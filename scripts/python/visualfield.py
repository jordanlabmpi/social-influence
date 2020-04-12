import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

from raycasting import *
from scipy.signal import savgol_filter

sys.path.append('/home/paul/Documents/multiviewtracks/MultiViewTracks/')
# download at https://github.com/pnuehrenberg/multiviewtracks and edit path accordingly

from tracks import tracks_to_pooled

def get_direction(tracks, heading=False):
    for i in tracks['IDENTITIES']:
        if heading:
            direction = np.arctan2(tracks[str(i)]['SPINE'][:, 2, 0] - tracks[str(i)]['SPINE'][:, 3, 0],
                                   tracks[str(i)]['SPINE'][:, 2, 1] - tracks[str(i)]['SPINE'][:, 3, 1])
            tracks[str(i)]['DIRECTION'] = direction
        else:
            direction = np.arctan2(np.diff(tracks[str(i)]['Y']), np.diff(tracks[str(i)]['X']))
            direction = np.append(direction, direction[-1])
            direction = savgol_filter(np.unwrap(direction), 31, 1)
            tracks[str(i)]['DIRECTION'] = wrap(direction)
    return tracks

def normalize_positions(positions, length=1000, padding=200):
    positions = positions - positions.min(axis=0)
    positions = (length - padding * 2) * positions / positions.max(axis=0)
    positions = positions + padding
    return positions

def contour_from_pose(spine, rad):

    unit_vectors = []
    for idx in range(1, len(spine)):
        unit_vectors.append(unit_vector(spine[idx - 1][::-1] - spine[idx][::-1]))

    mouth = np.array([spine[0, 1], spine[0, 0]]) + unit_vectors[0] * rad[0]
    tail = np.array([spine[-1, 1], spine[-1, 0]]) - unit_vectors[-1] * rad[-1]
    head_l = np.array([spine[0, 1], spine[0, 0]]) \
             + angle_to_vector(np.arctan2(spine[1, 0] - spine[2, 0],
                                          spine[1, 1] - spine[2, 1]) + np.pi / 2) * rad[0]
    head_r = np.array([spine[0, 1], spine[0, 0]]) \
             - angle_to_vector(np.arctan2(spine[1, 0] - spine[2, 0],
                                          spine[1, 1] - spine[2, 1]) + np.pi / 2) * rad[0]
    tail_l = np.array([spine[-1, 1], spine[-1, 0]]) \
             - angle_to_vector(np.arctan2(spine[-1, 0] - spine[-2, 0],
                                          spine[-1, 1] - spine[-2, 1]) + np.pi / 2) * rad[-1]
    tail_r = np.array([spine[-1, 1], spine[-1, 0]]) \
             + angle_to_vector(np.arctan2(spine[-1, 0] - spine[-2, 0],
                                          spine[-1, 1] - spine[-2, 1]) + np.pi / 2) * rad[-1]

    head = [mouth]
    tail = [tail_l, tail, tail_r]
    left = [head_l]
    right = [head_r]

    for to_head, to_tail, r, pt in zip(unit_vectors[:-1], unit_vectors[1:], rad[1:-1], spine[1:-1, ::-1]):
        if directed_angle(to_head, to_tail) < 0:
            side = -unit_vector(to_head - to_tail) * r
        else:
            side = unit_vector(to_head - to_tail) * r
        left_side = pt + side
        right_side = pt - side

        if magnitude(left_side - left[-1]) + magnitude(right_side - right[-1]) > \
           magnitude(left_side - right[-1]) + magnitude(right_side - left[-1]):
            left.append(right_side)
            right.append(left_side)
        else:
            left.append(left_side)
            right.append(right_side)

    contour = head + left + tail + right[::-1] + head
    contour = np.array(contour)
    contour = contour[np.isfinite(contour).all(axis=1)]

    unit_vectors = np.array(unit_vectors)[np.isfinite(unit_vectors).all(axis=1)]

    eyes = np.array([(contour[2] + contour[1]) / 2 + rotate_90_ccw(unit_vectors[0]) * 2,
                     (contour[-3] + contour[-2]) / 2 - rotate_90_ccw(unit_vectors[0]) * 2])

    return contour, eyes

def get_individuals(pooled, frame, active):
    positions = np.transpose([pooled['SPINE'][pooled['FRAME_IDX'] == frame, 1, 1],
                              pooled['SPINE'][pooled['FRAME_IDX'] == frame, 1, 0]])
    spines = pooled['SPINE'][pooled['FRAME_IDX'] == frame]
    radii = pooled['RADII'][pooled['FRAME_IDX'] == frame]
    headings = pooled['DIRECTION'][pooled['FRAME_IDX'] == frame]
    ids = pooled['IDENTITY'][pooled['FRAME_IDX'] == frame].tolist()

    individuals = []
    eye_list = []
    focal = None
    focal_heading = None
    focal_position = None
    for identity, pos, heading, spine, rad in zip(ids, positions, headings, spines, radii):
        eyes = []
        if identity == active:
            focal = []
            focal_heading = heading
            focal_position = pos
        fish, eyes = contour_from_pose(spine, rad)
        if identity == active:
            focal = [eye for eye in eyes]
        eyes = np.array([tuple(eye) for eye in eyes])
        eye_list.append(eyes)
        individuals.append(fish)
    return focal, focal_heading, focal_position, eye_list, individuals, ids

# functions for fish test

from copy import deepcopy

def get_visual_field(tracks, frames=None, active=None, visualize=True):

    data_segments = []
    data_ids = []
    data_visual_field = []
    data_position = []

    def mouse_click(event, x, y, flags, param):
        nonlocal individuals, idx, active
        if event == cv2.EVENT_LBUTTONDOWN:
            active = None
            for idx, fish in enumerate(individuals):
                if -cv2.pointPolygonTest(fish.reshape(-1, 1, 2).astype(np.int), (x, y), True) < 30:
                    active = ids[idx]

    tracks = get_direction(tracks, heading=True)
    pooled = tracks_to_pooled(tracks)

    pos_x = 500
    pos_y = 500

    individuals = []
    ids = []

    if visualize:
        window_name = 'visual field'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_click)
    for frame in tracks['FRAME_IDX'] if frames is None else frames:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)

        focal, focal_heading, focal_position, eye_list, individuals, ids = get_individuals(pooled, frame, active)

        data_position.append(focal_position)

        visible_intersects = []
        visible_ids = []

        if active is not None and focal is not None:
            focal = np.array(focal) # , dtype=np.int
            individuals.append(np.array([(0, 0), (0, 720), (1280, 720), (1280, 0)]).reshape(-1, 2))
            ids.append(-1)
            segments, segment_ids = collect_segments(individuals, ids)
            vertices, vertice_ids = collect_vertices(individuals, ids)

            visual_canvas = [img.copy(), img.copy()]
            data_visual_field.append([])

            for idx, eye in enumerate(focal):
                segment_rays = collect_segment_rays(segments, eye)
                vertice_rays = collect_vertice_rays(segments, eye)
                vertice_rays = np.sort(np.concatenate([vertice_rays - 10e-6,
                                                       vertice_rays + 10e-6]))

                intersects, intersect_ids = cast_rays(eye, vertice_rays, segments, segment_rays, segment_ids)

                visible_intersects.append(intersects)
                visible_ids.append(intersect_ids)

                contour = np.array(intersects).reshape(-1, 1, 2) # .astype(np.int)
                contour = [np.insert(contour, -1, contour[0], axis=0)]
                data_visual_field[-1].append(contour)

                if visualize:
                    cv2.drawContours(visual_canvas[idx], [np.array(intersects).reshape(-1, 1, 2).astype(np.int)], -1, (100, 100, 100), -1, cv2.LINE_AA)

            visual_canvas = cv2.addWeighted(visual_canvas[0], 0.5, visual_canvas[1], 0.5, 0)
            img = cv2.addWeighted(img, 0.5, visual_canvas, 0.5, 0)

        if visualize:
            if active is None:
                draw_objects(img, individuals, ids, [])
            else:
                draw_objects(img, individuals, ids, [-1])
            for eyes in eye_list:
                for eye in eyes.astype(np.int):
                    cv2.circle(img, tuple(eye), 2, (100, 100, 100), -1, cv2.LINE_AA)
            if focal is not None:
                for eye in focal.astype(np.int):
                    cv2.circle(img, tuple(eye), 2, (207, 255, 244), -1, cv2.LINE_AA)

        if len(visible_intersects) == 2 and active is not None:
            exclude_ids = [-1]
            visible_intersects = np.concatenate(visible_intersects, axis=0)
            visible_ids = np.concatenate(visible_ids, axis=0)
            visible_intersects = visible_intersects[visible_ids != active]
            visible_ids = visible_ids[visible_ids != active]
            visible_intersects = visible_intersects[np.invert(np.isin(visible_ids, exclude_ids))]
            visible_ids = visible_ids[np.invert(np.isin(visible_ids, exclude_ids))]

            visible_segments, visible_segment_ids = collect_visible_segments(visible_intersects,
                                                                             visible_ids,
                                                                             focal_position,
                                                                             focal_heading)
            data_segments.append(visible_segments)
            data_ids.append(visible_segment_ids)

            if visualize:
                for segment in visible_segments:
                    segment = np.array(segment)
                    cv2.polylines(img, [segment.reshape(-1, 1, 2).astype(np.int)], False, (207, 255, 244), 1, cv2.LINE_AA)

        if visualize:
            cv2.imshow(window_name, img)
            k = cv2.waitKey(18 if frames is None or len(frames) > 1 else 0) & 0xFF
            if k == 27:
                break
    if visualize:
        cv2.destroyWindow(window_name)

    return data_segments, data_ids, data_visual_field, data_position
