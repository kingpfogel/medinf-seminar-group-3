import sys
sys.path.append("C:/Users/Kai/PycharmProjects/MedInf/SepsisPrediction-1-master/src/sepsis_prediction_lstm")

from sepsis_pred_grid import run

epochs = [10, 20, 30]
lr = [0.01, 0.005, 0.001]
emb_size = [8, 16, 24]
hidden_size = [4, 6, 8]
num_layers = [2, 4, 6]

for i in range(0, len(epochs)):
    for j in range(0, len(lr)):
        for k in range(0, len(emb_size)):
            for m in range(0, len(hidden_size)):
                for p in range(0, len(num_layers)):
                    out = "C:\\Users\\Kai\\PycharmProjects\\MedInf\\SepsisPrediction-1-master\\out_12_6_GridSearch_\\" + str(epochs[i]) + "_" + str(lr[j]) + "_" + str(emb_size[k]) + "_" + str(hidden_size[m]) + "_" + str(num_layers[p]) + "\\best_model\\"
                    run(out, epochs[i], lr[j], emb_size[k], hidden_size[m], num_layers[p])
