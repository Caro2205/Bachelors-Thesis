import torch


def main(): 
    
    path = f'CoreLSTM/testing_predictions/train_loss/6_15_200_MSELoss()_0.01_(0.9, 0.999)_0.0_40_2000_nse2e-05_corrVel.pt'
    


    with torch.no_grad():

        data = torch.load(path)
        print(data[0])

        print(data[-1])

        
        





if __name__ == "__main__":
    main()