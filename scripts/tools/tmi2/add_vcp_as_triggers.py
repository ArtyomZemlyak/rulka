"""
This script reads a .npy file containing a list of VCP, and connects to a TMInterface instance to add the VCP as triggers.

This script would typically be used to check that the .npy file contains Virtual CheckPoints (VCP) that are properly placed.
"""

import argparse
from pathlib import Path

import numpy as np

from trackmania_rl.tmi_interaction.tminterface2 import MessageType, TMInterface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", type=Path)
    parser.add_argument("--tmi_port", "-p", type=int, default=8477)
    parser.add_argument("--timeout", "-t", type=int, default=60, help="Timeout in seconds (default: 60)")
    args = parser.parse_args()

    iface = TMInterface(args.tmi_port)
    vcp = np.load(args.npy_path)

    if not iface.registered:
        while True:
            try:
                print(f"Connecting to TMInterface on port {args.tmi_port} with timeout {args.timeout}s...")
                iface.register(args.timeout)
                break
            except ConnectionRefusedError as e:
                print(f"Connection refused: {e}")
                print("Make sure TMInterface is running on the specified port!")
                return

    print("Waiting for messages from TMInterface...")
    print("Make sure the game is running and a map is loaded!")
    print("You may need to start a race (press Enter in the game)")
    
    try:
        while True:
            msgtype = iface._read_int32()
            # =============================================
            #        READ INCOMING MESSAGES
            # =============================================
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = iface._read_int32()
                # ============================
                # BEGIN ON RUN STEP
                # ============================
                # ============================
                # END ON RUN STEP
                # ============================
                iface._respond_to_call(msgtype)
            elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                current = iface._read_int32()
                target = iface._read_int32()
                # ============================
                # BEGIN ON CP COUNT
                # ============================
                # ============================
                # END ON CP COUNT
                # ============================
                iface._respond_to_call(msgtype)
            elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
                iface._read_int32()
                iface._respond_to_call(msgtype)
            elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                iface._respond_to_call(msgtype)
            elif msgtype == int(MessageType.C_SHUTDOWN):
                iface.close()
            elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                print(f"Adding {len(vcp)} virtual checkpoints as triggers (every 10th point)...")
                for i in range(0, len(vcp), 10):
                    iface.execute_command(
                        f"add_trigger {vcp[i][0] - 0.4:.2f} {vcp[i][1] - 0.4:.2f} {vcp[i][2] - 0.4:.2f} {vcp[i][0] + 0.4:.2f} {vcp[i][1] + 0.4:.2f} {vcp[i][2] + 0.4:.2f}"
                    )
                print(f"Successfully added {len(range(0, len(vcp), 10))} triggers!")
                print("You can now see the triggers in the game. Press Ctrl+C to exit.")
                iface._respond_to_call(msgtype)
            else:
                pass
    except TimeoutError as e:
        print(f"\n❌ Timeout error: {e}")
        print("\nPossible reasons:")
        print("  1. Game is not running or is in the menu")
        print("  2. No map is loaded in the game")
        print("  3. Race has not been started (you need to press Enter in the game)")
        print("  4. TMInterface lost connection with the game")
        print("\nTroubleshooting:")
        print("  - Make sure the game is running")
        print("  - Load a map in the game")
        print("  - Start a race (press Enter)")
        print("  - Try increasing timeout with --timeout parameter (e.g., --timeout 120)")
    except KeyboardInterrupt:
        print("\n\nExiting...")
        if iface.registered:
            iface.close()


if __name__ == "__main__":
    main()
