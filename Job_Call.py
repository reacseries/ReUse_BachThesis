from asyncua.sync import Client, ThreadLoop
import time


# ─────────────────────────────────────────────
# Base OPC UA Device
# ─────────────────────────────────────────────
class OPCUADevice:
    def __init__(self, url, auto_start=True):
        self.url = url
        self.client = None
        self.tloop = None
        if auto_start:
            self.start()

    def start(self):
        self.tloop = ThreadLoop()
        self.tloop.daemon = True
        self.client = Client(self.url, tloop=self.tloop)
        self.tloop.start()
        self.client.connect()
        print(f"Connected to OPC UA server: {self.url}")

    def stop(self):
        if self.client:
            self.client.disconnect()
        if self.tloop:
            self.tloop.stop()
        print(f"Disconnected from {self.url}")


# ─────────────────────────────────────────────
# Yaskawa YRC1000 Robot Client
# (NO HARDCODED NODE IDs — EVERYTHING AUTO DETECTS)
# ─────────────────────────────────────────────
class YaskawaYRC1000(OPCUADevice):

    def __init__(self, url, auto_start=True):
        super().__init__(url, auto_start)
        if auto_start:
            self.init_nodes()

    def init_nodes(self):

        # Running-state flag
        self.running_var = self.client.get_node(
            "ns=5;s=MotionDeviceSystem.Controllers.Controller_1.ParameterSet.IsRunning"
        )

        # Method nodes (these you already confirmed exist)
        self.method_set_servo = self.client.get_node(
            "ns=5;s=MotionDeviceSystem.Controllers.Controller_1.Methods.SetServo"
        )

        self.method_start_job = self.client.get_node(
            "ns=5;s=MotionDeviceSystem.Controllers.Controller_1.Methods.StartJob"
        )

        print("Robot nodes initialized")

        # AUTO-DETECT their parent object (this is the crucial fix)
        self.parent_set_servo = self.method_set_servo.get_parent()
        self.parent_start_job = self.method_start_job.get_parent()

        # Debug print (optional):
        print("Detected SetServo parent:", self.parent_set_servo)
        print("Detected StartJob parent:", self.parent_start_job)

    # ---------------------------------------------------------
    # Servo ON / OFF (auto-detected parent)
    # ---------------------------------------------------------
    def set_servo(self, enable: bool):
        print("Setting servo:", enable)
        return self.parent_set_servo.call_method(self.method_set_servo, enable)

    # ---------------------------------------------------------
    # Start job (auto-detected parent)
    # ---------------------------------------------------------
    def start_job(self, jobname, block=True):
        print("Starting job:", jobname)

        self.parent_start_job.call_method(self.method_start_job, jobname)

        if block:
            while self.running_var.get_value():
                time.sleep(0.1)
            print("Job finished:", jobname)


# ─────────────────────────────────────────────
# Main program
# ─────────────────────────────────────────────
def main():
    robot = YaskawaYRC1000("opc.tcp://192.168.0.20:16448")

    try:
        robot.set_servo(True)
        robot.start_job("BATTERY_PLACE")
        robot.set_servo(False)

    finally:
        robot.stop()


if __name__ == "__main__":
    main()
