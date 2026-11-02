import time
import airsim


client = airsim.MultirotorClient()

print("Connecting to AirSim...")
client.confirmConnection()

print("Enabling API control...")
client.enableApiControl(True)

print("Arming...")
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()

time.sleep(1)

state = client.getMultirotorState()
kin = state.kinematics_estimated

print("\nCurrent state:")
print("position =", kin.position)
print("velocity =", kin.linear_velocity)
print("orientation =", kin.orientation)
print("acceleration =", kin.linear_acceleration)

print("\nFlying forward for 2 seconds...")
client.moveByVelocityAsync(
    1.0,     # vx
    0.0,     # vy
    0.0,     # vz
    duration=2.0
).join()

print("Hovering...")
client.hoverAsync().join()

time.sleep(2)

print("Landing...")
client.landAsync().join()

client.armDisarm(False)
client.enableApiControl(False)

print("Finished.")