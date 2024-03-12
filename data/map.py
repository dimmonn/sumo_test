city_map = {
    'intersections': {
        'A': {'neighbors': {'B': 10, 'C': 5}},
        'B': {'neighbors': {'A': 10, 'D': 15}},
        'C': {'neighbors': {'A': 5, 'D': 10}},
        'D': {'neighbors': {'B': 15, 'C': 10}}
    },
    'traffic_conditions': {
        'A-B': 'light',
        'A-C': 'congested',
        'B-D': 'normal',
        'C-D': 'accident'
    }
}

rewards = {
    'positive': 100,
    'negative': -10,
    'waiting_penalty': -5
}
