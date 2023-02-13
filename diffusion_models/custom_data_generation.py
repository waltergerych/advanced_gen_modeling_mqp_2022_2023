import random


# Define file extension name
CUSTOM_TYPE = "small_noise"

# Generate data into array
data = []

for i in range (200):
    # low_to_high_gradual
    # data.append((1/200)*i)

    # high_to_low_gradual
    # data.append(1 - ((1/200)*i))

    # small_noise
    data.append(random.uniform(0.1, 0.3))

print(data)


with open(f'../dataset/UCI_HAR_Dataset_Triaxial/train/Inertial_Signals/custom_acc_{CUSTOM_TYPE}.txt', 'w') as f:
    f.write(' '.join(map(str, data)))
    f.write("\n")
    f.write(' '.join(map(str, data)))
