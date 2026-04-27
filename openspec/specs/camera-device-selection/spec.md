# Purpose
Provides user interface and logic for enumerating, selecting, and initializing camera devices for live mode operation.

## ADDED

### FR1: Device enumeration
- CameraSelector lists all available video capture devices using cv2.VideoCapture
- Returns list of device IDs and available resolutions

### FR2: Device selection UI
- GUI dropdown populated with enumerated devices
- CLI prompts user to select device by number

### FR3: Camera initialization
- Selected camera opened with default resolution (1280x720 or device default)
- Camera configuration set (exposure, focus if supported)

### FR4: Error handling
- Graceful handling of camera busy/permission denied
- User prompted to select different device on failure

## MODIFIED

### N/A

## REMOVED

### N/A
