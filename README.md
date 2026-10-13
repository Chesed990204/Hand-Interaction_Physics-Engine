# Hand Tracking Tutorial — Unreal Engine

An Unreal Engine project for exploring hand interaction and Python-to-Unreal data streaming. It includes a hand-interaction level, Blueprint assets, hand-related input actions, and a small TCP example that sends synthetic 3D coordinates to the Unreal Editor.

**Unreal Engine 5.3.2 · Blueprints · Meta XR · Python**

## What to explore

- **Hand interaction:** inspect the `HandInteraction` level, `VRPawn`, ball, and end-effector Blueprints.
- **Input setup:** examine the default input mapping context and action assets for grasp, index curl, pointing, and thumbs-up gestures.
- **External data input:** run a Python TCP server and receive its three-value messages inside the Unreal Editor.

The TCP example generates random coordinates. It provides a minimal communication example; it does not stream measured hand positions.

## Requirements

| Component | Purpose |
| --- | --- |
| Unreal Engine **5.3.2** | Version specified by the original project README; the `.uproject` associates with UE 5.3. |
| Meta XR plugin compatible with UE 5.3 | The project enables the plugin named `OculusXR`. Plugin files are not bundled. |
| Git and Git LFS | Required to retrieve the actual content files. |
| A compatible Meta Quest setup | For inspecting hand/controller interaction in VR. |
| Python 3 | For the standalone TCP server; it uses only standard-library modules. |
| Unreal's Python Editor Script Plugin | For running `tcp_client.py` inside the editor. |

## Get the project

All files under `Content/`, **including the Python scripts**, are tracked through Git LFS.

```bash
git lfs install
git clone https://github.com/Chesed990204/HandTrackingTutorial.git
cd HandTrackingTutorial
git lfs pull
```

If a downloaded file contains `version https://git-lfs.github.com/spec/v1` instead of its actual contents, the LFS objects have not been retrieved. Resolve that before opening the project or running the scripts.

## Open the hand-interaction level

1. Install Unreal Engine **5.3.2** and a compatible **Meta XR / OculusXR** plugin.
2. Open `HandTrackingTutorial.uproject`.
3. In the Content Browser, open:

   ```text
   Content/HandInteraction.umap
   ```

   The committed game default map is Unreal's `OpenWorld` template, so select `HandInteraction` explicitly. Set it as the startup/game map in **Project Settings → Maps & Modes** if needed.

4. Inspect `VRPawn`, `VRPawnGameMode`, and `IMC_Default` to follow the pawn and input setup. The project uses **Enhanced Input** and configures Meta XR for **Controllers and Hands**.
5. Configure a compatible headset connection and XR runtime before trying **VR Preview**. The committed configuration targets desktop hardware and also contains Android/Quest settings; a fresh headset run or packaged build has not been verified for this README.

## Run the TCP example

This example uses two processes on the same computer:

```text
Python server → TCP 127.0.0.1:12345 → Unreal Editor Python client
```

### 1. Start the server

From the repository root:

```bash
python Content/Python/tcp_server.py
```

The server sends three values approximately every 0.1 seconds. Each value is a random integer from -10 to 9, encoded as a float using `struct.pack('fff', ...)`.

### 2. Start the editor client

Enable **Python Editor Script Plugin** under **Edit → Plugins** and restart the editor. In the Output Log's Python console, run:

```python
import tcp_client
tcp_client.start_streaming_in_thread()
```

Unreal adds the project's `Content/Python` directory to its Python search path. The committed Python settings also contain a machine-specific additional path; update or remove that entry for your checkout. See [Epic's UE 5.3 Python setup guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/scripting-the-unreal-editor-using-python?application_version=5.3).

The client stores the latest received vector in `UnrealDataStore` and logs updates. To inspect it:

```python
tcp_client.UnrealDataStore.get_float_values()
```

Start the client once per session. Stop the external server with **Ctrl+C**.

### Current example behavior

- The default client path **stores values**; it does not move an actor.
- A separate actor-application function is present, but its `set_actor_location` call is commented out and its target actor name is hard-coded.
- The receiver assumes that `recv(12)` returns a complete three-float message. A more general streaming implementation needs to accumulate partial reads and define byte order explicitly.
- `tcp_client.py` imports `unreal` and belongs in the **Unreal Editor**. It is not a standalone Python client or a packaged-game networking implementation.

## Project structure

| Path | Contents |
| --- | --- |
| `HandTrackingTutorial.uproject` | Engine association and plugin requirements. |
| `Content/HandInteraction.umap` | Hand-interaction level. |
| `Content/Blueprints/VRPawn.uasset` | VR pawn Blueprint. |
| `Content/Blueprints/BP_Ball.uasset` | Ball Blueprint. |
| `Content/Blueprints/BP_endeffector.uasset` | End-effector Blueprint. |
| `Content/Blueprints/Input/` | Mapping context and hand/controller action assets. |
| `Content/Blueprints/Networking/` | Networking Blueprint asset. |
| `Content/Python/tcp_server.py` | Synthetic coordinate stream. |
| `Content/Python/tcp_client.py` | Editor-side TCP receiver and vector storage. |
| `Config/` | Input, XR, map, and Python settings. |
