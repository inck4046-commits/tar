BUFFS_ALL = {
    '0벞':              {'hp': 0.0, 'atk': 0.0, 'def': 0.0},
    'HP20%':        {'hp': 0.2, 'atk': 0.0, 'def': 0.0},
    'ATK20%':       {'hp': 0.0, 'atk': 0.2, 'def': 0.0},
    'DEF20%':       {'hp': 0.0, 'atk': 0.0, 'def': 0.2},
    'HP40%':        {'hp': 0.4, 'atk': 0.0, 'def': 0.0},
    'ATK40%':       {'hp': 0.0, 'atk': 0.4, 'def': 0.0},
    'DEF40%':       {'hp': 0.0, 'atk': 0.0, 'def': 0.4},
    'HP+ATK20%':    {'hp': 0.2, 'atk': 0.2, 'def': 0.0},
    'HP+DEF20%':    {'hp': 0.2, 'atk': 0.0, 'def': 0.2},
    'ATK+DEF20%':   {'hp': 0.0, 'atk': 0.2, 'def': 0.2},
}
ALL_BUFFS = list(BUFFS_ALL.items())
ONE_BUFFS = [
    ('HP20%', BUFFS_ALL['HP20%']),
    ('ATK20%', BUFFS_ALL['ATK20%']),
    ('DEF20%', BUFFS_ALL['DEF20%']),
]

TAR_BUFFS = [
    ('HP40%', BUFFS_ALL['HP40%']),
    ('ATK40%', BUFFS_ALL['ATK40%']),
    ('DEF40%', BUFFS_ALL['DEF40%']),
    ('HP+ATK20%', BUFFS_ALL['HP+ATK20%']),
    ('HP+DEF20%', BUFFS_ALL['HP+DEF20%']),
    ('ATK+DEF20%', BUFFS_ALL['ATK+DEF20%']),
]





