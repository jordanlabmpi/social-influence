import numpy as np
import numba
import cv2

def densify_objects(objects):
    densified_objects = []
    for obj in objects:
        obj_x = np.interp(np.linspace(0, 1, obj.shape[0] * 5), np.linspace(0, 1, obj.shape[0]), obj[:, 0])
        obj_y = np.interp(np.linspace(0, 1, obj.shape[0] * 5), np.linspace(0, 1, obj.shape[0]), obj[:, 1])
        obj = np.transpose([obj_x, obj_y])
        densified_objects.append(obj)
    return densified_objects

def draw_objects(img, objects, object_ids, exclude_ids, color=(100, 100, 100), fill=False):
    for obj, obj_id in zip(objects, object_ids):
        if obj_id not in exclude_ids:
            if not fill:
                cv2.polylines(img, [obj.reshape(-1, 1, 2).astype(np.int)], True, color, 1, cv2.LINE_AA)
            else:
                cv2.drawContours(img, [obj.reshape(-1, 1, 2).astype(np.int)], -1, color, -1, cv2.LINE_AA)

@numba.jit(nopython=True)
def magnitude(vector):
    mag = 0
    for value in vector:
        mag += value ** 2
    return mag ** (1 / 2)

@numba.jit(nopython=True, cache=True)
def unit_vector(vector):
    return vector / magnitude(vector)

@numba.jit(nopython=True, cache=True)
def rotate_90_ccw(vector):
    return np.array([-vector[1], vector[0]])

@numba.jit(nopython=True, cache=True)
def directed_angle(u_vec_1, u_vec_2):
    angle = np.arccos(np.dot(u_vec_1, u_vec_2))
    u_vec_perp = rotate_90_ccw(u_vec_1)
    dot_perp = np.dot(u_vec_perp, u_vec_2)
    if dot_perp > 0:
        angle = -angle
    return angle

@numba.jit(nopython=True, cache=True)
def collect_segment_rays(segments, position):
    segments_shifted = segments.copy()
    segments_shifted[:, :2] = segments[:, :2] - position
    segments_shifted[:, 2:] = segments[:, 2:] - position
    rays = vectors_to_angle(segments_shifted.reshape(-1, 2)).reshape(-1, 2)
    return rays

@numba.jit(nopython=True, cache=True)
def collect_vertice_rays(vertices, position):
    vertices_shifted = vertices.copy()
    vertices_shifted[:, :2] = vertices[:, :2] - position
    rays = vectors_to_angle(vertices_shifted)
    return rays

@numba.jit(nopython=True, cache=True)
def wrap(angles):
    return (angles + np.pi) % (2 * np.pi ) - np.pi

@numba.jit(nopython=True, cache=True)
def parallel(vector_1, vector_2):
    return np.abs(vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0]) == 0

@numba.jit(nopython=True, cache=True)
def vector_to_angle(vector):
    return np.arctan2(vector[1], vector[0])

@numba.jit(nopython=True, cache=True)
def vectors_to_angle(vectors):
    return np.arctan2(vectors[:, 1], vectors[:, 0])

@numba.jit(nopython=True, cache=True)
def angle_to_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])

@numba.jit(nopython=True, cache=True)
def get_intersect(point, ray, segment):
    intersect = np.array([np.nan, np.nan])
    ray_x, ray_y = point
    ray_dx, ray_dy = angle_to_vector(ray)
    segment_x = segment[0]
    segment_y = segment[1]
    segment_dx = segment[2] - segment[0]
    segment_dy = segment[3] - segment[1]
    if not parallel((ray_dx, ray_dy), (segment_dx, segment_dy)):
        T2 = (ray_dx * (segment_y - ray_y) + ray_dy * (ray_x - segment_x)) / (segment_dx * ray_dy - segment_dy * ray_dx)
        if ray_dx != 0:
            T1 = (segment_x + segment_dx * T2 - ray_x) / ray_dx
        else:
            T1 = (segment_y + segment_dy * T2 - ray_y) / ray_dy
        if T1 > 0 and 0 < T2 < 1:
            intersect = np.array([ray_x + ray_dx * T1, ray_y + ray_dy * T1])
    return intersect

@numba.jit(nopython=True, cache=True)
def check_ray(ray_lower, ray_upper, ray):
    if ray_lower < ray_upper:
        if ray_upper - ray_lower < np.pi:
            return ray_lower < ray < ray_upper
        else:
            return ((ray < ray_lower) and (ray + 2 * np.pi > ray_upper)) or \
                   ((ray - 2 * np.pi < ray_lower) and (ray > ray_upper))
    else:
        if ray_lower - ray_upper < np.pi:
            return ray_upper < ray < ray_lower
        else:
            return ((ray < ray_upper) and (ray + 2 * np.pi > ray_lower)) or \
                   ((ray - 2 * np.pi < ray_upper) and (ray > ray_lower))

@numba.jit(nopython=True, cache=True)
def cast_ray(point, ray, segments, segment_rays, segment_ids):
    nearest_intersect = np.array([np.nan, np.nan])
    nearest_distance = np.inf
    nearest_id = np.nan
    for idx in range(segments.shape[0]):
        ray_lower = segment_rays[idx][0]
        ray_upper = segment_rays[idx][1]
        if check_ray(ray_lower, ray_upper, ray):
            intersect = get_intersect(point, ray, segments[idx])
            distance = distance_point_to_point(point, intersect)
            if distance < nearest_distance:
                nearest_intersect = intersect
                nearest_distance = distance
                nearest_id = segment_ids[idx]
    return nearest_intersect, nearest_id

@numba.jit(nopython=True, cache=True)
def cast_rays(point, rays, segments, segment_rays, segment_ids):
    intersects = []
    intersect_ids = []
    for ray in rays:
        intersect, intersect_id = cast_ray(point, ray, segments, segment_rays, segment_ids)
        if not np.isnan(intersect).any():
            intersects.append(intersect)
            intersect_ids.append(intersect_id)
    return intersects, intersect_ids

@numba.jit(nopython=True, cache=True)
def distance_point_to_point(point_1, point_2=np.array([0, 0])):
    return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** (1 / 2)

@numba.jit(nopython=True, cache=True)
def distances_to_point(array, point=np.array([0, 0])):
    return np.sqrt(np.square(array - point).sum(axis=1))

@numba.jit(nopython=True, cache=True)
def collect_visible_segments(visible_intersects, visible_ids, focal_position, focal_heading):
    visible_segments = []
    visible_segment_ids = []
    visible_rays = wrap(collect_vertice_rays(visible_intersects, focal_position) - focal_heading)
    sort = np.argsort(visible_rays)
    visible_intersects = visible_intersects[sort]
    visible_rays = visible_rays[sort]
    visible_ids = visible_ids[sort]
    previous_id = None
    segment = []
    segment_ids = []
    for idx, visible_id in enumerate(visible_ids):
        if previous_id is None:
            segment.append(visible_intersects[idx])
            segment_ids.append(visible_id)
        elif visible_id != previous_id:
            if len(segment) > 0:
                visible_segments.append(segment)
                visible_segment_ids.append(segment_ids)
            segment = [visible_intersects[idx]]
            segment_ids = [visible_id]
        else:
            segment.append(visible_intersects[idx])
            segment_ids.append(visible_id)
        previous_id = visible_id
    visible_segments.append(segment)
    visible_segment_ids.append(segment_ids)
    if visible_segment_ids[0][0] == visible_segment_ids[-1][0]: # need to add interrupted segment that crosses -pi/pi
        visible_segments.append([visible_segments[-1][-1], visible_segments[0][0]])
        visible_segment_ids.append([visible_segment_ids[0][0]] * 2)
    return visible_segments, visible_segment_ids

def collect_segments(objects, object_ids, exclude_ids=[]):
    segments = np.hstack([np.concatenate([obj for obj, obj_id in zip(objects, object_ids) if obj_id not in exclude_ids]),
                          np.concatenate([np.roll(obj, 1, axis=0) for obj, obj_id in zip(objects, object_ids) if obj_id not in exclude_ids])])
    segment_ids = np.concatenate([np.repeat(obj_id, obj.shape[0]) for obj, obj_id in zip(objects, object_ids) if obj_id not in exclude_ids])
    return segments, segment_ids

def collect_vertices(objects, object_ids, exclude_ids=[]):
    vertices = np.concatenate([obj for obj, obj_id in zip(objects, object_ids) if obj_id not in exclude_ids])
    vertice_ids = np.concatenate([np.repeat(obj_id, obj.shape[0]) for obj, obj_id in zip(objects, object_ids) if obj_id not in exclude_ids])
    return vertices, vertice_ids

if __name__ == '__main__':
    # create objects for interactive test
    # (contours are closed automatically, first should not be the same as last)

    thing_1 = cv2.ellipse2Poly((200, 200), (80, 140), 0, 0, 360, 10)[1:]
    thing_2 = cv2.ellipse2Poly((800, 600), (120, 40), 40, 0, 360, 10)[1:]
    thing_3 = np.array([(800, 180), (860, 240), (700, 150), (660, 360), (610, 80), (720, 120)]).reshape(-1, 2)
    thing_4 = np.array([(400, 700), (450, 700), (400, 750)]).reshape(-1, 2)
    border_1 = np.array([(50, 50), (50, 950), (950, 950), (950, 50)]).reshape(-1, 2)
    border_2 = np.array([(0, 0), (0, 1000), (1000, 1000), (1000, 0)]).reshape(-1, 2)

    objects = [thing_1, thing_2, thing_3, thing_4]
    objects = densify_objects(objects)
    objects.extend([border_1, border_2])
    object_ids = [1, 2, 3, 4, -1, -2]
    exclude_ids = []

    # functions for interactive test

    def mouse_move(event, x, y, flags, param):
        global pos_x, pos_y
        if event == cv2.EVENT_MOUSEMOVE:
            pos_x, pos_y = x, y

    def move(position, heading, speed=None):
        global pos_x, pos_y
        heading_towards = unit_vector(np.array([pos_x, pos_y]) - position)
        if np.isfinite(heading_towards).all():
            if speed is None:
                position = np.array([pos_x, pos_y])
            else:
                position = position + heading_towards * speed
            heading = heading_towards
        return position, heading

    # initialize and run the interactive test, leave by pressing ESC

    position = np.array([500, 500], dtype=np.float)
    heading = np.array([1, 0])

    pos_x = 500
    pos_y = 500

    window_name = 'visual field'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_move)

    while True:
        img = np.zeros((1000, 1000 ,3), dtype=np.uint8)

        segments, segment_ids = collect_segments(objects, object_ids)
        vertices, vertice_ids = collect_vertices(objects, object_ids)
        segment_rays = collect_segment_rays(segments, position)
        vertice_rays = collect_vertice_rays(segments, position)
        vertice_rays = np.sort(np.concatenate([vertice_rays - 10e-5,
                                               vertice_rays + 10e-5]))

        intersects, intersect_ids = cast_rays(position, vertice_rays, segments, segment_rays, segment_ids)

        contour = [np.array(intersects).reshape(-1, 1, 2).astype(np.int)]
        cv2.drawContours(img, contour, -1, (50, 50, 50), -1, cv2.LINE_AA)

        draw_objects(img, objects, object_ids, exclude_ids)

        intersects = np.array(intersects)[np.invert(np.isin(intersect_ids, exclude_ids))]
        intersect_ids = np.array(intersect_ids)[np.invert(np.isin(intersect_ids, exclude_ids))]
        visible_segments, visible_segment_ids = collect_visible_segments(intersects,
                                                                         intersect_ids,
                                                                         position,
                                                                         vector_to_angle(heading))

        for segment in visible_segments:
            segment = np.round(segment, 0)
            cv2.polylines(img, [segment.reshape(-1, 1, 2).astype(np.int)], False, (207, 255, 244), 1, cv2.LINE_AA)

        cv2.circle(img, tuple(position.astype(np.int)), 5, (250, 250, 250), -1, cv2.LINE_AA)
        arrow = tuple((position - 20 * heading).astype(np.int))
        cv2.line(img, tuple(position.astype(np.int)), arrow, (250, 250, 250), 1, cv2.LINE_AA)

        cv2.imshow(window_name, img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyWindow(window_name)
            break

        objects = [thing_1, thing_2, thing_3, thing_4]
        objects = densify_objects(objects)
        objects.extend([border_1, border_2])
        object_ids = [1, 2, 3, 4, -1, -2]
        exclude_ids = []

        position, heading = move(position, heading)
