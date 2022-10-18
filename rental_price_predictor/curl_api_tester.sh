curl -i -H "Content-Type: application/json" -X 'POST' -d '[
  {
    "model_key": "Citroën",
    "mileage": 100000,
    "engine_power": 100,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "convertible",
    "private_parking_available": true,
    "has_gps": true,
    "has_air_conditioning": true,
    "automatic_car": true,
    "has_getaround_connect": true,
    "has_speed_regulator": true,
    "winter_tires": true
  },
  {
    "model_key": "Audi",
    "mileage": 100000,
    "engine_power": 150,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "convertible",
    "private_parking_available": true,
    "has_gps": true,
    "has_air_conditioning": true,
    "automatic_car": true,
    "has_getaround_connect": true,
    "has_speed_regulator": true,
    "winter_tires": true
  }
]' https://acsts-getaround-price-predict.herokuapp.com/predict