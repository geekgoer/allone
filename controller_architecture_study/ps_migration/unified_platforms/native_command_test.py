
def test_packaged_native_command_mapping_expiry_and_postcommand_state():
    from ps_core.controls import control_to_airsim_command
    io = NativeCommandIO(['Drone1'], ThrustCalibration(), .003, mock=True, allow_late=False)
    try:
        row = plan(time.perf_counter())
        io.send(np.array([30., 4., -4., 3.]), row)
        packet, done = collect(io, 'drain')[0]
        assert done['control'] == [20., 3., -3., 2.]
        cmd = control_to_airsim_command(done['control'], ThrustCalibration(), 1.)
        # The full Python API source and three wire fixtures are audited separately.
        wire = [cmd.roll_rate, -cmd.pitch_rate, -cmd.yaw_rate, cmd.throttle]
        assert done['native_rpc_arguments'] == pytest.approx(wire)
        assert packet[2] >= done['rpc_return_s'] and packet[0].quaternion[0] == 1.
        row = plan(time.perf_counter(), number=1)
        row['deadline_s'] = time.perf_counter()-1.
        io.send(np.array([9.81, 0., 0., 0.]), row)
        packet, done = collect(io, 'drain')[0]
        assert packet is None and done['status'] == 'expired_not_applied'
        assert 'rpc_return_s' not in done
    finally:
        io.close()
