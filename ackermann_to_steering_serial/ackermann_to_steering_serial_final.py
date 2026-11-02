#!/usr/bin/python3

import csv
import sys
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
from diagnostic_msgs.msg import KeyValue
from std_msgs.msg import Bool, Float32MultiArray
import serial

# Steering board states, as reported in the last column of the DATA line.
STATE_OFF = 0      # PID off, DAC idle
STATE_TRACK = 1    # PID follows s<deg> targets
STATE_HOLD = 2     # PID holds the angle it had when hold started, s<deg> ignored

STATE_CMD = {STATE_OFF: "d", STATE_TRACK: "e", STATE_HOLD: "x"}

KEEPALIVE_PERIOD = 0.1   # s, "h" to the board (board holds after 300 ms of silence)
DATA_TIMEOUT = 0.3       # s, board prints DATA every 50 ms
STATE_GRACE = 0.3        # s, time the board gets to confirm a state command
REOPEN_PERIOD = 1.0      # s, retry interval after the port is lost


class AckermannToSteeringSerial(Node):
    def __init__(self, serial_port='/dev/ttyUSB1', baud_rate=115200):
        super().__init__('ackermann_to_steering_serial')

        self.serial_port = serial_port
        self.baud_rate = baud_rate

        if not serial_port.startswith('/dev/serial/by-id/'):
            self.get_logger().warn(
                f'{serial_port} can swap with the sterfBoard port on boot; '
                f'pass the /dev/serial/by-id/... path instead (ls -l /dev/serial/by-id/)'
            )

        # Safety state.
        # flag: last /signal_error key from mode_switch (0 disarmed, 1 armed ok, 2 armed + error).
        # fault: latched steering fault, reported on /steering_status. Starts set and is
        # cleared only while disarmed with a healthy link, so recovery is disarm -> arm.
        self.flag = 0
        self.fault = True
        self.fault_reason = 'Startup'
        self.last_data_time = 0.0
        self.last_state_cmd_time = 0.0
        self.last_reopen_try = 0.0
        self.ino_state = STATE_OFF

        # Opening the port resets the board; it reports READY and DATA lines ~1-2 s later.
        self.serial_connection = None
        self.open_serial()

        self.last_target = None
        self.last_send_time = 0.0
        self._last_debug_print_time = 0.0
        self._last_raw_print_time = 0.0

        self.min_send_period = 0.02
        self.min_target_step = 0.2

        self.last_cmd_target = 0.0
        self.last_cmd_rate = 0.0
        self.last_cmd_speed = 0.0
        self.last_cmd_acc = 0.0
        self.last_cmd_jerk = 0.0

        self.rc_steer_raw = 0.0
        self.rc_speed_raw = 0.0
        self.rc_target_rate = 0.0
        self.rc_manual_target = 0.0

        self.ino_t_ms = 0
        self.ino_target_deg = 0.0
        self.ino_target_slew_deg = 0.0
        self.ino_angle_deg = 0.0
        self.ino_error_deg = 0.0
        self.ino_delta_v = 0.0
        self.ino_va_out = 0.0
        self.ino_vb_out = 0.0
        self.ino_codeA = 0
        self.ino_codeB = 0
        self.ino_count = 0
        self.ino_pid_on = 0
        self.ino_p_term = 0.0
        self.ino_i_term = 0.0
        self.ino_d_term = 0.0
        self.ino_isr = 0
        self.ino_invalid = 0

        self.ackermann_subscriber = self.create_subscription(
            AckermannDrive,
            'ackermann_cmd',
            self.ackermann_callback,
            10
        )

        self.rc_debug_subscriber = self.create_subscription(
            Float32MultiArray,
            'rc_steer_debug',
            self.rc_debug_callback,
            10
        )

        self.signal_error_subscriber = self.create_subscription(
            KeyValue,
            'signal_error',
            self.signal_flag_callback,
            10
        )

        # Steering feedback; mode_switch uses data[0] as the measured angle.
        # Important: this is NOT /feedback_data, so it cannot disturb old mode_switch.
        self.steering_feedback_publisher = self.create_publisher(
            Float32MultiArray,
            'steering_feedback',
            10
        )

        # True while the board link is healthy and no steering fault is latched.
        self.steering_status_publisher = self.create_publisher(
            Bool,
            'steering_status',
            10
        )

        self.create_timer(0.01, self.read_serial)
        self.create_timer(0.02, self.log_row)
        self.create_timer(KEEPALIVE_PERIOD, self.supervise)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = open(f'steering_test_log_{ts}.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'pc_time',
            'rc_steer_raw',
            'rc_speed_raw',
            'rc_target_rate_deg_s',
            'rc_manual_target_deg',
            'cmd_target_deg',
            'cmd_target_rate_deg_s',
            'cmd_speed_m_s',
            'cmd_acc_m_s2',
            'cmd_jerk',
            'ino_t_ms',
            'ino_target_deg',
            'ino_target_slew_deg',
            'ino_angle_deg',
            'ino_error_deg',
            'ino_delta_v',
            'ino_va_out',
            'ino_vb_out',
            'ino_codeA',
            'ino_codeB',
            'ino_count',
            'ino_pid_on',
            'ino_p_term',
            'ino_i_term',
            'ino_d_term',
            'ino_isr',
            'ino_invalid',
            'ino_state',
            'flag',
            'steering_ok',
        ])
        self.csv_file.flush()

    def open_serial(self):
        self.last_reopen_try = time.time()
        try:
            self.serial_connection = serial.Serial(self.serial_port, self.baud_rate, timeout=0.02)
        except (serial.SerialException, OSError) as e:
            self.serial_connection = None
            self.set_fault(f'Cannot open {self.serial_port}: {e}')
            return
        self.ino_state = STATE_OFF
        self.get_logger().info(f'Connected to steering board on {self.serial_port}')

    def close_serial(self, reason):
        self.set_fault(reason)
        try:
            self.serial_connection.close()
        except Exception:
            pass
        self.serial_connection = None

    def set_fault(self, reason):
        # Keep the first reason; follow-up symptoms of the same fault are not logged.
        if not self.fault:
            self.get_logger().error(f'STEERING FAULT: {reason}')
            self.fault = True
            self.fault_reason = reason

    def link_ok(self):
        return (self.serial_connection is not None and
                time.time() - self.last_data_time <= DATA_TIMEOUT)

    def desired_state(self):
        if self.flag == 0:
            return STATE_OFF
        if self.flag == 1 and not self.fault:
            return STATE_TRACK
        # Emergency, or a fault mode_switch has not reacted to yet: stop where it is.
        return STATE_HOLD

    def send_state_cmd(self, state):
        self.send_line(STATE_CMD[state])
        self.last_state_cmd_time = time.time()

    def send_line(self, line: str):
        if self.serial_connection is None:
            return
        msg = line.strip() + "\n"
        try:
            self.serial_connection.write(msg.encode('ascii', errors='ignore'))
        except (serial.SerialException, OSError) as e:
            self.close_serial(f'Serial write failed: {e}')

    def signal_flag_callback(self, msg: KeyValue):
        flag = int(msg.key) if msg.key in ("0", "1", "2") else 2
        if flag == self.flag:
            return
        self.flag = flag
        self.send_state_cmd(self.desired_state())

    def supervise(self):
        now = time.time()

        if self.serial_connection is None:
            if now - self.last_reopen_try >= REOPEN_PERIOD:
                self.open_serial()
        else:
            self.send_line("h")

        if self.serial_connection is not None and now - self.last_data_time > DATA_TIMEOUT:
            self.set_fault('No DATA from steering board')

        desired = self.desired_state()
        if (self.link_ok() and self.ino_state != desired and
                now - self.last_state_cmd_time > STATE_GRACE):
            if desired == STATE_TRACK:
                # Board left tracking on its own (e.g. its serial timeout put it in hold).
                self.set_fault(f'Board state {self.ino_state}, expected tracking')
            else:
                self.send_state_cmd(desired)

        if self.fault and self.flag == 0 and self.link_ok():
            self.fault = False
            self.get_logger().info(f'Steering fault cleared (was: {self.fault_reason})')

        status = Bool()
        status.data = self.link_ok() and not self.fault
        self.steering_status_publisher.publish(status)

    def ackermann_callback(self, msg: AckermannDrive):
        target = float(msg.steering_angle)
        target = max(-620.0, min(620.0, target))

        self.last_cmd_target = target
        self.last_cmd_rate = float(msg.steering_angle_velocity)
        self.last_cmd_speed = float(msg.speed)
        self.last_cmd_acc = float(msg.acceleration)
        self.last_cmd_jerk = float(msg.jerk)

        # Targets only go out while armed, fault free and the board is tracking.
        if self.desired_state() != STATE_TRACK or self.ino_state != STATE_TRACK:
            self.last_target = None
            return

        now = time.time()
        should_send = False

        if self.last_target is None:
            should_send = True
        elif abs(target - self.last_target) >= self.min_target_step:
            should_send = True
        elif (now - self.last_send_time) >= self.min_send_period:
            should_send = True

        if should_send:
            self.send_line(f"s{target:.2f}")

            if (now - self._last_debug_print_time) >= 0.10:
                self._last_debug_print_time = now
                self.get_logger().info(
                    f'Sent steering target: {target:.2f} deg, '
                    f'rate={self.last_cmd_rate:.2f} deg/s'
                )

            self.last_target = target
            self.last_send_time = now

    def rc_debug_callback(self, msg: Float32MultiArray):
        if len(msg.data) >= 4:
            self.rc_steer_raw = float(msg.data[0])
            self.rc_speed_raw = float(msg.data[1])
            self.rc_target_rate = float(msg.data[2])
            self.rc_manual_target = float(msg.data[3])

    def read_serial(self):
        if self.serial_connection is None:
            return

        try:
            while self.serial_connection.in_waiting > 0:
                line = self.serial_connection.readline().decode(
                    'ascii',
                    errors='ignore'
                ).strip()

                if not line:
                    continue

                if line == "READY":
                    # Board rebooted (port open, watchdog, brown-out or reset button).
                    # Its encoder zero is now wherever the wheel was at boot.
                    self.ino_state = STATE_OFF
                    self.set_fault('Steering board reset')
                    self.get_logger().info("Steering board raw: READY")
                    continue

                if line.startswith("DATA,"):
                    parts = line.split(',')

                    if len(parts) >= 19:
                        try:
                            t_ms = int(parts[1])
                            if t_ms < self.ino_t_ms:
                                self.ino_state = STATE_OFF
                                self.set_fault('Steering board reset (time went backwards)')
                            self.ino_t_ms = t_ms
                            self.ino_target_deg = float(parts[2])
                            self.ino_target_slew_deg = float(parts[3])
                            self.ino_angle_deg = float(parts[4])
                            self.ino_error_deg = float(parts[5])
                            self.ino_delta_v = float(parts[6])
                            self.ino_va_out = float(parts[7])
                            self.ino_vb_out = float(parts[8])
                            self.ino_codeA = int(parts[9])
                            self.ino_codeB = int(parts[10])
                            self.ino_count = int(parts[11])
                            self.ino_pid_on = int(parts[12])
                            self.ino_p_term = float(parts[13])
                            self.ino_i_term = float(parts[14])
                            self.ino_d_term = float(parts[15])
                            self.ino_isr = int(parts[16])
                            self.ino_invalid = int(parts[17])
                            self.ino_state = int(parts[18])
                            self.last_data_time = time.time()

                            fb = Float32MultiArray()
                            fb.data = [
                                self.ino_angle_deg,
                                self.ino_target_deg,
                                self.ino_error_deg,
                                float(self.ino_pid_on),
                                self.ino_va_out,
                                self.ino_vb_out,
                                self.ino_delta_v,
                                float(self.ino_count),
                                float(self.ino_invalid),
                            ]
                            self.steering_feedback_publisher.publish(fb)

                        except ValueError:
                            pass

                else:
                    # Print INFO/WARN occasionally without flooding.
                    now = time.time()
                    if line.startswith(("INFO", "WARN", "ERR")) or (now - self._last_raw_print_time) >= 0.5:
                        self._last_raw_print_time = now
                        self.get_logger().info(f"Steering board raw: {line}")

        except (serial.SerialException, OSError) as e:
            self.close_serial(f'Serial read failed: {e}')
        except Exception as e:
            self.get_logger().warn(f"Serial read warning: {e}")

    def log_row(self):
        self.csv_writer.writerow([
            time.time(),
            self.rc_steer_raw,
            self.rc_speed_raw,
            self.rc_target_rate,
            self.rc_manual_target,
            self.last_cmd_target,
            self.last_cmd_rate,
            self.last_cmd_speed,
            self.last_cmd_acc,
            self.last_cmd_jerk,
            self.ino_t_ms,
            self.ino_target_deg,
            self.ino_target_slew_deg,
            self.ino_angle_deg,
            self.ino_error_deg,
            self.ino_delta_v,
            self.ino_va_out,
            self.ino_vb_out,
            self.ino_codeA,
            self.ino_codeB,
            self.ino_count,
            self.ino_pid_on,
            self.ino_p_term,
            self.ino_i_term,
            self.ino_d_term,
            self.ino_isr,
            self.ino_invalid,
            self.ino_state,
            self.flag,
            int(self.link_ok() and not self.fault),
        ])
        self.csv_file.flush()

    def destroy_node(self):
        self.send_line("d")

        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass

        if self.serial_connection is not None:
            try:
                self.serial_connection.close()
            except Exception:
                pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    serial_port = '/dev/ttyUSB1'
    baud_rate = 115200

    if args is None:
        args = sys.argv

    if len(args) >= 2:
        serial_port = args[1]

    node = None

    try:
        node = AckermannToSteeringSerial(
            serial_port=serial_port,
            baud_rate=baud_rate
        )
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)
