import random


# Define file extension name
CUSTOM_TYPE = "three_values"

# Generate data into array
data = []

for i in range (200):
    # low_to_high_gradual
    # data.append((1/200)*i)

    # high_to_low_gradual
    # data.append(1 - ((1/200)*i))

    # small_noise
    # data.append(random.uniform(0.1, 0.3))

    # large_noise
    # data.append(random.uniform(0.1, 0.9))

    # mixed_noise
    # data.append(random.uniform(0.1, 0.3) if i < 100 else random.uniform(0.1, 0.9))

    # static_halves
    # data.append(0.1 if i < 100 else 0.9)

    # three_values
    if i < 200/3:
        data.append(0.1)
    elif i > 200/3 and i < (200/3)*2:
        data.append(0.5)
    else:
        data.append(0.9)

print(data)


with open(f'dataset/UCI_HAR_Dataset_Triaxial/train/Inertial_Signals/custom_acc_{CUSTOM_TYPE}.txt', 'w') as f:
    f.write(' '.join(map(str, data)))
    f.write("\n")
    f.write(' '.join(map(str, data)))
