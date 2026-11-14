# STL geometry + camera homography bundle

The STL and `lid_geometry_config.json` come from the exact uploaded mesh, SHA-256:
`aa9a014fccec7894804fb59b10930b924d018cc15f0aede608f880bf2a6663a1`.

Both uploaded copies matched this checksum. STL units are **assumed** mm because of the filename; STL files do not record units. Mesh scale to meters = 0.001 once verified.

## Transferring STL from Windows to Claude Code server

**Option 1: VS Code Remote-SSH**: Open the connected remote project; drag the local STL into the project's `aaltonomous/scene/assets/` folder in VS Code Explorer. Verify with `ls -l` and `sha256sum` on the server. Do not assume remote Claude Code can access Windows `C:\...` paths.

**Option 2: PowerShell SCP** (replace `USERNAME` and `SERVER` with your actual SSH account and host):

```powershell
scp "C:\Users\Edward\Downloads\lid_shape_holes_200mm.stl" USERNAME@SERVER:/tmp/lid_shape_holes_200mm.stl
```

Then run on your server (adjust `REPO` if yours differs):

```bash
REPO=/root/long/aaltonomous-humanoid-integration
mkdir -p "$REPO/aaltonomous/scene/assets"
cp /tmp/lid_shape_holes_200mm.stl "$REPO/aaltonomous/scene/assets/"
sha256sum "$REPO/aaltonomous/scene/assets/lid_shape_holes_200mm.stl"
```

If Claude's runtime is a container or VM different from your SSH login, copy the file into that runtime's actual accessible worktree. Ask Claude to run `realpath` on the file before editing any code.

## Geometry in STL-local frame

- XYZ bounds: [-100,-100,0] to [100,100,12] (assumed mm)
- Main plate: Z=0..4; rim reaches Z=12 (mesh orientation is not guaranteed to equal installed lid-up orientation)
- Square opening centroid: (-48,-35) mm; x bounds -78..-18, y bounds -65..-5
- Circle opening centroid: (48,-35) mm; radius approximately 30 mm
- Triangle opening polygon centroid: (0.005988,39.269655) mm; bounding rectangle -29.990953..29.990953, y 17.699701..72.300301
- All exact sampled section outlines are in `lid_geometry_config.json`, measured at z=2.

**These are local STL coordinates, not Isaac robot/world coordinates.** The installed lid pose, orientation, robot transform, correct Z, and gripper clearances all still matter.

## Camera calibration: calculate H only after measuring corners

The config DOES NOT contain a numeric homography, because one cannot be calculated from STL geometry alone. It needs **four matched pixel locations from the actual camera frame**, ideally from an undistorted image.

Copy `camera_corners_template.json` to `camera_corners.json` and fill `pixel_corners` with camera pixel coordinates `[u,v]` of the named corners:

- `minus_x_minus_y` -> STL (-100,-100)
- `plus_x_minus_y` -> STL (+100,-100)
- `plus_x_plus_y` -> STL (+100,+100)
- `minus_x_plus_y` -> STL (-100,+100)

**WARNING:** These names are STL corner labels, not necessarily camera top-left, top-right etc. Use the 3-hole layout to verify orientation. The outer perimeter is rounded: detect/fix the corresponding nominal corners using intersections of the four outer straight edges if necessary. Avoid swapping corner labels or mirroring the lid.

Install numpy if needed; no OpenCV required to calculate the 3×3 homography:

```bash
python calibrate_lid_homography.py --config lid_geometry_config.json --corners camera_corners.json --output lid_homography.json
```

The script prints `H_pixel_to_stl_xy_mm` and predicted image pixel centers for all holes. A homography maps points **on the lid plane only**. Camera lens distortion should be corrected first. Four-point fitting will always produce nearly zero corner reprojection error, so validate with additional independent hole-center or marker observations.

For Isaac Sim, if the lid's USD `worldTransform` is known, direct local->world transformations may be more reliable than a camera homography. In either case, a geometrically correct target does NOT guarantee the robot arm can reach it.
