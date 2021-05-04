"""
Feature Extraction & Matching Implementation.
Based on OpenCV Python samples at https://github.com/opencv/opencv/blob/master/samples/python/find_obj.py.
Interactive features were removed for maximum performance.
main() method was reserved for test use, may be deprecated in the future

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn
"""

import numpy as np
import cv2 as cv

from utilities import Timer

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH = 6


def init_feature(name):
    chunks = name.split('-')

    if chunks[0] == 'sift':
        detector = cv.SIFT_create()
        norm = cv.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv.xfeatures2d.SURF_create(800)
        norm = cv.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv.ORB_create(400)
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv.AKAZE_create()
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv.BRISK_create()
        norm = cv.NORM_HAMMING
    else:
        return None, None   # Return None if unknown detector name

    if 'flann' in chunks:
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2

        matcher = cv.FlannBasedMatcher(flann_params)
    else:
        matcher = cv.BFMatcher(norm)

    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []

    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])

    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)

    return p1, p2, list(kp_pairs)


def draw_match(result_title, img1, img2, kp_pairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create visualized result image
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            color = green
            cv.circle(vis, (x1, y1), 2, color, -1)
            cv.circle(vis, (x2, y2), 2, color, -1)
        else:
            color = red
            r = 2
            thickness = 3
            cv.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), color, thickness)
            cv.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), color, thickness)
            cv.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), color, thickness)
            cv.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), color, thickness)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv.line(vis, (x1, y1), (x2, y2), green)

    cv.imshow(result_title, vis)

    return vis


def main():
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')

    try:
        fn1, fn2 = args
    except:
        fn1 = 'box.png'
        fn2 = 'box_in_scene.png'

    img1 = cv.imread("/Users/langzhou/Downloads/input_0.png", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("/Users/langzhou/Downloads/input_1.png", cv.IMREAD_GRAYSCALE)
    detector, matcher = init_feature(feature_name)

    if img1 is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)

    if img2 is None:
        print('Failed to load fn2:', fn2)
        sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)

    print('using', feature_name)

    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

    with Timer("Matching..."):
        raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2

    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    if len(p1) >= 4:
        H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
        print(f"{np.sum(status)} / {len(status)}  inliers/matched")
    else:
        H, status = None, None
        print(f"{len(p1)} matches found, not enough for homography estimation")

    _vis = draw_match("Result", img1, img2, kp_pairs, status, H)

    cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
