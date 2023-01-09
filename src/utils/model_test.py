import torch

# 예측 테스트
def model_valid(model, scaler, validX_tensor, validY_tensor):
    with torch.no_grad():
        val_pred = []
        for pr in range(len(validX_tensor)):
            model.reset_hidden_state()

            predicted = model(torch.unsqueeze(validX_tensor[pr], 0))
            predicted = torch.flatten(predicted)
            val_pred.append(predicted)

        val_pred = torch.stack(val_pred, 0)

        # INVERSE
        val_pred_inverse = scaler.inverse_transform(val_pred.detach().cpu())
        validY_inverse = scaler.inverse_transform(validY_tensor.detach().cpu())

    return val_pred_inverse, validY_inverse


def model_test(model, scaler, testX_tensor):
    with torch.no_grad():
        test_pred = []

        for pr in range(len(testX_tensor)):
            model.reset_hidden_state()

            predicted = model(torch.unsqueeze(testX_tensor[pr], 0))
            predicted = torch.flatten(predicted)
            test_pred.append(predicted)

        test_pred = torch.stack(test_pred, 0)

        # INVERSE
        test_pred_inverse = scaler.inverse_transform(test_pred.detach().cpu())

    return test_pred_inverse
