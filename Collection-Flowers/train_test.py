d = 0.3 # This is the distance decay factor (from the center), represented as lowercase delta in paper
# This class is used for the foveation algorithm
class Sample:
    def __init__(self, x, prefix_sum, x1, y1):
        self.x = x;
        self.x1 = x1
        self.y1 = y1
        self.diff = 0.0
        for i in self.x:
          self.diff += np.std(i)
        self.dist = math.sqrt(((95 - x1)**2) + ((95 - y1)**2))
        self.key = prefix_sum - (self.dist * d) + (3800 * self.diff)

    def __lt__(self, other):
        return self.key > other.key


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    maxLoss = 0
    for batch, (X, y) in enumerate(dataloader):
      # X will be in shape (1, 3, 224, 224), as mini_batch_size=1, there are 3 color channels, and images are 224x224 pixels
      # Firstly, we will create the reference array for 2D Prefix Sum
      z = X.squeeze()

      total_avg_arr = torch.zeros(224, 224)
      for z_arr in z:
          avg_arr = torch.abs(torch.mean(z_arr, dtype=torch.float) - z_arr)
          total_avg_arr += avg_arr

      ref_arr = torch.cumsum(torch.cumsum(avg_arr, dim=0), dim=1)
      arr = []

      z = z.numpy()
      ref_arr = ref_arr.numpy()
      for i_loop in range(4):
          for j_loop in range(4):
              i = 40*i_loop
              j = 40*j_loop
              prefix_sum = ref_arr[i+63][j+63] - (0 if i == 0 else ref_arr[i-1][j+63]) - (0 if j == 0 else ref_arr[i+63][j-1]) + (0 if (i == 0 or j == 0) else ref_arr[i-1][j-1])
              arr.append(Sample(np.array([z[0][i:i+64, j:j+64], z[1][i:i+64, j:j+64], z[2][i:i+64, j:j+64]]), prefix_sum, i, j))

      arr.sort()
      coll_data = []
      for i in range(10):
        coll_data.append(hsv_to_rgb(arr[i].x.transpose(1, 2, 0)).transpose(2, 0, 1))

      np_data = np.array(coll_data)
      model_input_data = torch.reshape(torch.tensor(np_data), (10, 3, 64, 64)).to(device) # The input data, in shape [10, 3, 64, 64]

      sample_size = 10 # Manually setting this for now

      # We can now run the prediction
      pred = model(model_input_data) # In shape [10, 102]

      # Answer, we will use y[0] though
      y = y.to(device)

      # To calculate the loss, we will sum it up using the following loos:
      time_decay = 35 # This is how slowly the loss function decreases in importance with respect to time
      t = 0
      prev_x, prev_y = 0,0
      loss = torch.tensor(0, dtype=torch.float).to(device) # This is the loss, we add to it

      for i in range(10):
        sample = arr[i]
        if t == 0:
          prev_x = sample.x1
          prev_y = sample.y1

        dt = math.sqrt(((prev_x - sample.x1)**2) + ((prev_y - sample.y1)**2)) / 128000
        prev_x, prev_y = sample.x1, sample.y1

        # Actual training
        # Note: We do the optimization step at the end, so we average the gradients, which is done by multiplying the loss by 1/n
        multiplier = max(0, (1/6) * (math.log((-t/time_decay) + 1, 1.5) + 1) * ((1 + dt) ** (1 + dt)))
        loss += loss_fn(pred[i], y[0]) * multiplier
        t += 1

      # As we received all the gradients, do the optimization step
      loss.backward() # This will cumulate the gradients
      optimizer.step()
      optimizer.zero_grad()

      if batch % 600 == 0:
          loss, current = loss.item(), batch + len(X)
          print(f"{loss:>7f}, {current:>5d}")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
          # We will "collect" the outputs
          # Essentially, we will average the predictions for each sample
          pred = torch.zeros((1, 102)).to(device)

          z = X.squeeze()

          total_avg_arr = torch.zeros(224, 224)
          for z_arr in z:
              avg_arr = torch.abs(torch.mean(z_arr, dtype=torch.float) - z_arr)
              total_avg_arr += avg_arr

          ref_arr = torch.cumsum(torch.cumsum(avg_arr, dim=0), dim=1)
          arr = []

          z = z.numpy()
          ref_arr = ref_arr.numpy()
          for i_loop in range(4):
              for j_loop in range(4):
                  i = 40*i_loop
                  j = 40*j_loop
                  prefix_sum = ref_arr[i+63][j+63] - (0 if i == 0 else ref_arr[i-1][j+63]) - (0 if j == 0 else ref_arr[i+63][j-1]) + (0 if (i == 0 or j == 0) else ref_arr[i-1][j-1])
                  arr.append(Sample(np.array([z[0][i:i+64, j:j+64], z[1][i:i+64, j:j+64], z[2][i:i+64, j:j+64]]), prefix_sum, i, j))

          arr.sort()
          coll_data = []
          for i in range(10):
            coll_data.append(hsv_to_rgb(arr[i].x.transpose(1, 2, 0)).transpose(2, 0, 1))

          np_data = np.array(coll_data)
          model_input_data = torch.reshape(torch.tensor(np_data), (10, 3, 64, 64)).to(device) # The input data, in shape [10, 3, 64, 64]

          sample_size = 10 # Manually setting this for now

          # We can now run the prediction
          pred = model(model_input_data) # In shape [10, 102]

          # Answer, we will use y[0] though
          y = y.to(device)

          # Now, we will find the average array to multiply pred by
          # This array will be in shape [10, 1]
          mult = []
          t = 0
          time_decay = 35
          prev_x, prev_y = 0,0
          for i in range(10):
            sample = arr[i]
            if t == 0:
              prev_x = sample.x1
              prev_y = sample.y1

            dt = math.sqrt(((prev_x - sample.x1)**2) + ((prev_y - sample.y1)**2)) / 128000
            prev_x, prev_y = sample.x1, sample.y1

            #pred += model(x, t) * (1/sample_size) * 1.5 * (0.8**t) * ((1 + dt) ** (1 + dt))
            temp = []
            temp.append(max(0, (1/6) * (math.log((-t/time_decay) + 1, 1.5) + 1) * ((1 + dt) ** (1 + dt))))
            mult.append(temp)

            t += 1

          multT = torch.tensor(mult).to(device)
          pred = pred * multT
          actual_pred = torch.mean(pred, dim=0, keepdim=True, dtype=torch.float) # Shape is in [1, 102]
          test_loss += loss_fn(actual_pred, y).item()
          correct += (actual_pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"{100*correct:>0.1f}%, {test_loss:>8f}\n")
