#!/bin/sh

# >>> Parking occupancy <<<
# Mobile net v2
tensorman run --gpu --python3 python -- ./train.py -y True --model MobileNetV2 --output parking\ models/mobile_net_v2/50_mobile_net_v2 --data-dir parking_dataset --epochs 50 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model MobileNetV2 --output parking\ models/mobile_net_v2/40_mobile_net_v2 --data-dir parking_dataset --epochs 40 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model MobileNetV2 --output parking\ models/mobile_net_v2/30_mobile_net_v2 --data-dir parking_dataset --epochs 30 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model MobileNetV2 --output parking\ models/mobile_net_v2/20_mobile_net_v2 --data-dir parking_dataset --epochs 20 --classes 43

# Inception Residual Network V2
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionResNetV2 --output parking\ models/inception_res_net_v2/50_inception_res_net_v2 --data-dir parking_dataset --epochs 50 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionResNetV2 --output parking\ models/inception_res_net_v2/40_inception_res_net_v2 --data-dir parking_dataset --epochs 40 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionResNetV2 --output parking\ models/inception_res_net_v2/30_inception_res_net_v2 --data-dir parking_dataset --epochs 30 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionResNetV2 --output parking\ models/inception_res_net_v2/20_inception_res_net_v2 --data-dir parking_dataset --epochs 20 --classes 43

# Inception V3
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionV3 --output parking\ models/inception_v3/50_inception_v3 --data-dir parking_dataset --epochs 50 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionV3 --output parking\ models/inception_v3/40_inception_v3 --data-dir parking_dataset --epochs 40 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionV3 --output parking\ models/inception_v3/30_inception_v3 --data-dir parking_dataset --epochs 30 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionV3 --output parking\ models/inception_v3/20_inception_v3 --data-dir parking_dataset --epochs 20 --classes 43


# >>> German traffic signs <<<
# Mobile net v2
tensorman run --gpu --python3 python -- ./train.py -y True --model MobileNetV2 --output german\ traffic\ signs\ models/mobile_net_v2/50_mobile_net_v2 --data-dir german_traffic_signs --epochs 50 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model MobileNetV2 --output german\ traffic\ signs\ models/mobile_net_v2/40_mobile_net_v2 --data-dir german_traffic_signs --epochs 40 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model MobileNetV2 --output german\ traffic\ signs\ models/mobile_net_v2/30_mobile_net_v2 --data-dir german_traffic_signs --epochs 30 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model MobileNetV2 --output german\ traffic\ signs\ models/mobile_net_v2/20_mobile_net_v2 --data-dir german_traffic_signs --epochs 20 --classes 43

# Inception Residual Network V2
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionResNetV2 --output german\ traffic\ signs\ models/inception_res_net_v2/50_inception_res_net_v2 --data-dir german_traffic_signs --epochs 50 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionResNetV2 --output german\ traffic\ signs\ models/inception_res_net_v2/40_inception_res_net_v2 --data-dir german_traffic_signs --epochs 40 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionResNetV2 --output german\ traffic\ signs\ models/inception_res_net_v2/30_inception_res_net_v2 --data-dir german_traffic_signs --epochs 30 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionResNetV2 --output german\ traffic\ signs\ models/inception_res_net_v2/20_inception_res_net_v2 --data-dir german_traffic_signs --epochs 20 --classes 43

# Inception V3
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionV3 --output german\ traffic\ signs\ models/inception_v3/50_inception_v3 --data-dir german_traffic_signs --epochs 50 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionV3 --output german\ traffic\ signs\ models/inception_v3/40_inception_v3 --data-dir german_traffic_signs --epochs 40 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionV3 --output german\ traffic\ signs\ models/inception_v3/30_inception_v3 --data-dir german_traffic_signs --epochs 30 --classes 43
tensorman run --gpu --python3 python -- ./train.py -y True --model InceptionV3 --output german\ traffic\ signs\ models/inception_v3/20_inception_v3 --data-dir german_traffic_signs --epochs 20 --classes 43
