import time
import airsim


client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

T = 0.05

for k in range(100):  # 5 秒
    state = client.getMultirotorState()
    kin = state.kinematics_estimated

    p = kin.position
    v = kin.linear_velocity
    q = kin.orientation
    a = kin.linear_acceleration
    w = kin.angular_velocity

    print(
        f"k={k:03d} | "
        f"p=({p.x_val:.3f},{p.y_val:.3f},{p.z_val:.3f}) | "
        f"v=({v.x_val:.3f},{v.y_val:.3f},{v.z_val:.3f}) | "
        f"q=({q.w_val:.3f},{q.x_val:.3f},{q.y_val:.3f},{q.z_val:.3f})"
    )

    time.sleep(T)

client.hoverAsync().join()