## GNSS Global Constraints — NavSatFix → ENU → UTM

### Coordinate pipeline

The `gnss_global` module accepts `sensor_msgs/NavSatFix` directly and operates in **ENU** (East-North-Up) coordinates with origin at the first valid RTK fix. The pipeline is:

1. **ENU datum** — the first NavSatFix with `status >= min_fix_status` establishes the geodetic origin (lat/lon/alt). All subsequent messages are converted to ENU relative to that origin using standard WGS84 formulas.

2. **SVD alignment** — once `min_baseline` metres of trajectory have been accumulated, a 2D SVD computes the ENU→world rotation that aligns the LiDAR odometry with the GPS positions. `gnss_datum.json` is written at this moment — not earlier, because the heading rotation is not available until the SVD has fired.

3. **Position factor** — `gtsam::PoseTranslationPrior` on all 3 axes (X=East, Y=North, Z=Up) with dynamic sigma derived from the NavSatFix `position_covariance`.

### Quality gating

| Fix status | Behaviour |
|---|---|
| `STATUS_NO_FIX` (-1) | Rejected — no factor added |
| `STATUS_FIX` (0) | Rejected if `min_fix_status >= 1` |
| `STATUS_SBAS_FIX` (1) | Accepted with noise inflated by `sbas_noise_inflation` |
| `STATUS_GBAS_FIX` (2) | Accepted with nominal receiver sigma |

### Dynamic sigma

If `position_covariance_type != COVARIANCE_TYPE_UNKNOWN`, the per-axis sigma is derived directly from the receiver covariance (`sqrt(cov[0,4,8])`), with:
- **Floor**: 0.5 cm — values below floor are clamped with a warning (the Septentrio mosaic G5 typically reports 0.7–1 cm at RTK fix)
- **Cap**: 20 m — measurements with sigma > 20 m are rejected

If covariance is unavailable, `prior_inf_scale` from config is used as fallback.

### gnss_datum.json

Written to the dump directory at SVD alignment time. Contains:
- Geodetic coordinates of the ENU origin (lat/lon/alt)
- UTM zone and UTM coordinates of the origin
- `T_enu_world` transformation matrix (ENU→GLIM world frame rotation)

### UTM export

The offline viewer loads `gnss_datum.json` and exposes an **"Export in UTM coordinates"** checkbox in the PLY export dialog. When active, x/y/z coordinates are exported as `property double` in absolute UTM (zone locked to datum). If the file is not present, the checkbox is greyed out.

### Configuration parameters (`config_gnss_global.json`)

```json
{
  "gnss_topic": "/navsatfix",
  "min_fix_status": 1,
  "sbas_noise_inflation": 10.0,
  "prior_inf_scale": [1e3, 1e3, 1e3],
  "min_baseline": 10.0
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gnss_topic` | string | `/navsatfix` | ROS2 topic to subscribe to. Must be `sensor_msgs/NavSatFix`. |
| `min_fix_status` | int | `1` | Minimum NavSatFix status accepted. `0` = autonomous fix, `1` = SBAS/RTK float, `2` = RTK fix only. Measurements below this threshold are rejected entirely. |
| `sbas_noise_inflation` | double | `10.0` | Noise inflation factor applied when fix status is below `STATUS_GBAS_FIX` (RTK fix). The position sigma is multiplied by this value, making the GNSS factor softer during degraded signal — e.g. under bridges. Has no effect when RTK fix is active. |
| `prior_inf_scale` | [double x3] | `[1e3, 1e3, 1e3]` | Fallback precision (1/σ²) for East, North, Up axes used when the receiver does not report `position_covariance` (i.e. `covariance_type == COVARIANCE_TYPE_UNKNOWN`). Default value corresponds to σ ≈ 31 mm. When dynamic sigma is active (covariance available), this parameter is ignored. |
| `min_baseline` | double | `10.0` | Minimum trajectory length in metres before the SVD alignment fires. The GNSS module buffers measurements until this distance has been travelled, ensuring the SVD has enough geometric spread to compute a reliable ENU→world rotation. Setting this too low risks a poorly conditioned SVD with 180° heading ambiguity. |
