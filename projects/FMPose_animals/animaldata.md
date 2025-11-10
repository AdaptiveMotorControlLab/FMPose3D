
keypoint: 
            [0: 'left_eye', 1: 'right_eye', 2:'mouth_mid', 3:'left_front_paw',
            4: 'right_front_paw', 5:'left_back_paw', 6: 'right_back_paw', 7: 'tail_base',
            8: 'left_front_thigh', 9: 'right_front_thigh', 10: 'left_back_thigh', 11: 'right_back_thigh',
            12: 'left_shoulder', 13: 'right_shoulder', 14: 'left_front_knee', 15: 'right_front_knee',
            16: 'left_back_knee', 17: 'right_back_knee',18: 'neck', 19: 'tail_end',
            20: 'left_ear', 21: 'right_ear', 22: 'left_mouth', 23: 'right_mouth',
            24: 'nose', 25: 'tail_mid']

skeleton = [
    # --- Head connections ---
    (24, 0),   # nose → left_eye
    (24, 1),   # nose → right_eye
    (1, 21),   # right_eye right_ear 
    (0, 21),   # left_eye left_ear 
    (24, 2),   # nose → mouth_mid
    (2, 22),   # mouth_mid → left_mouth
    (2, 23),   # mouth_mid → right_mouth
    (24, 18),  # nose → neck

    # --- Upper body (neck/shoulders/spine/tail) ---
    (18, 12),  # neck → left_shoulder
    (18, 13),  # neck → right_shoulder
    (12, 8),   # left_shoulder → left_front_thigh
    (13, 9),   # right_shoulder → right_front_thigh
    (8, 14),   # left_front_thigh → left_front_knee
    (9, 15),   # right_front_thigh → right_front_knee
    (14, 3),   # left_front_knee → left_front_paw
    (15, 4),   # right_front_knee → right_front_paw

    # --- Spine and hind legs ---
    (18, 7),   # neck → tail_base
    (7, 10),   # tail_base → left_back_thigh
    (7, 11),   # tail_base → right_back_thigh
    (10, 16),  # left_back_thigh → left_back_knee
    (11, 17),  # right_back_thigh → right_back_knee
    (16, 5),   # left_back_knee → left_back_paw
    (17, 6),   # right_back_knee → right_back_paw

    # --- Tail ---
    (7, 25),   # tail_base → tail_mid
    (25, 19),  # tail_mid → tail_end
]