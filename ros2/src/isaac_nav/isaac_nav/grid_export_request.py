"""Asks the running Isaac Sim to export its occupancy grid, or to list what is near a point.

Both requests go over UDP to goal_display.py inside Isaac Sim. The export needs the
simulation playing. The file is written inside the Isaac Sim container; with the usual
``~/Volume:/Volume`` mount, ``/Volume/NavRL/maps/x.npz`` is ``~/Volume/NavRL/maps/x.npz``
on the host. Results appear in the Isaac Sim log (``[goal_display] ...``).
"""

import argparse
import json
import socket


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--x', type=float, nargs=2, default=(-20.0, 20.0), metavar=('MIN', 'MAX'))
    p.add_argument('--y', type=float, nargs=2, default=(-20.0, 20.0), metavar=('MIN', 'MAX'))
    p.add_argument('--z', type=float, nargs=2, default=(0.0, 5.0), metavar=('MIN', 'MAX'))
    p.add_argument('--resolution', type=float, default=0.5)
    p.add_argument('--clearance', type=float, default=0.6)
    p.add_argument('--probe', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                   help='instead of exporting, list the prims near this point')
    p.add_argument('--probe-radius', type=float, default=1.0)
    p.add_argument('--out', default='/Volume/NavRL/maps/scene_grid.npz',
                   help='path as seen from inside the Isaac Sim container')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=9870)
    a = p.parse_args()

    if a.probe:
        msg = {'cmd': 'probe', 'x': a.probe[0], 'y': a.probe[1], 'z': a.probe[2],
               'radius': a.probe_radius}
        note = 'Probe requested; see the Isaac Sim log for "[goal_display] probe ..."'
    else:
        # Accept the limits in either order.
        msg = {'cmd': 'export_grid', 'x_range': sorted(a.x), 'y_range': sorted(a.y),
               'z_range': sorted(a.z), 'resolution': a.resolution,
               'clearance': a.clearance, 'out_file': a.out}
        note = 'Export requested; watch the Isaac Sim log for "[goal_display] grid ..."'
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(
        json.dumps(msg).encode(), (a.host, a.port))
    print(note)


if __name__ == '__main__':
    main()
