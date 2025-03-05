d = 0.3 # This is the distance decay factor (from the center), represented as lowercase delta in paper
# This class is used for the foveation algorithm
class Sample:
    def __init__(self, x, prefix_sum, x1, y1):
        self.x = x;
        self.x1 = x1
        self.y1 = y1
        self.diff = np.std(x)
        self.dist = math.sqrt(((13 - x1)**2) + ((13 - y1)**2))
        self.key = prefix_sum - (self.dist * d) + (28 * self.diff)

    def __lt__(self, other):
        return self.key > other.key

# Train Loop
# If you want to train for n epochs, call this function n times
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
      # X will be in shape (1, 1, 28, 28), as mini_batch_size=1, there is 1 color channel, and images are 28x28 pixels
      # Firstly, we will create the reference array for 2D Prefix Sum
      z = X.squeeze()
      ref_arr = torch.cumsum(torch.cumsum(z, dim=0), dim=1)

      z = z.numpy()
      ref_arr = ref_arr.numpy()
      # Populate array for processing
      arr = []
      for i_loop in range(3):
        for j_loop in range(3):
            i = 7*i_loop
            j = 7*j_loop
            prefix_sum = ref_arr[i+13][j+13] - (0 if i == 0 else ref_arr[i-1][j+13]) - (0 if j == 0 else ref_arr[i+13][j-1]) + (0 if (i == 0 or j == 0) else ref_arr[i-1][j-1])
            arr.append(Sample(z[i:i+14, j:j+14], prefix_sum, i, j))

      arr.sort()
      coll_data = []
      for i in range(6):
        coll_data.append(arr[i].x)

      np_data = np.array(coll_data)
      model_input_data = torch.reshape(torch.tensor(np_data), (6, 1, 14, 14)).to(device) # The input data, in shape [6, 1, 14, 14]

      sample_size = 6 # The number of regions we will consider

      # We can now run the prediction
      pred = model(model_input_data) # In shape [6, 10]

      # Answer, we will use y[0] though to maintain shape
      y = y.to(device)

      # To calculate the loss, we will sum it up using the following:
      time_decay = 18 # This is how slowly the loss function decreases in importance concerning time, written as tau in the paper
      t = 0
      prev_x, prev_y = 0,0
      loss = torch.tensor(0, dtype=torch.float).to(device) # This is the loss, we add to it

      for i in range(sample_size):
        sample = arr[i]
        if t == 0:
          # This is the edge case, dt will be 0 so the entire term will have no effect
          prev_x = sample.x1
          prev_y = sample.y1

        dt = math.sqrt(((prev_x - sample.x1)**2) + ((prev_y - sample.y1)**2)) / 2800 # Here, nu is 1/2800
        prev_x, prev_y = sample.x1, sample.y1

        # Actual training
        # Note: We do the optimization step at the end, where we do the weighted average of the loss, which is done by multiplying each individual loss by the following
        multiplier = max(0, (1/3) * (math.log((-t/time_decay) + 1, 1.5) + 1) * ((1 + dt) ** (1 + dt)))
        loss += loss_fn(pred[i], y[0]) * multiplier
        t += 1

      # As we added the weighted sums of the loss to the total loss, do the optimization step
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

# Train Loop
# Prints accuracy
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
          # We will "collect" the outputs
          # Essentially, we will average the predictions for each sample according to their weights
          pred = torch.zeros((1, 10)).to(device)

          z = X.squeeze()
          ref_arr = torch.cumsum(torch.cumsum(z, dim=0), dim=1) # 2D Prefix Sum

          z = z.numpy()
          ref_arr = ref_arr.numpy()

          # Populate array for processing
          arr = []
          for i_loop in range(3):
            for j_loop in range(3):
              i = 7*i_loop
              j = 7*j_loop
              prefix_sum = ref_arr[i+13][j+13] - (0 if i == 0 else ref_arr[i-1][j+13]) - (0 if j == 0 else ref_arr[i+13][j-1]) + (0 if (i == 0 or j == 0) else ref_arr[i-1][j-1])
              arr.append(Sample(z[i:i+14, j:j+14], prefix_sum, i, j))

          arr.sort()
          coll_data = []
          for i in range(6):
            coll_data.append(arr[i].x)

          np_data = np.array(coll_data)
          model_input_data = torch.reshape(torch.tensor(np_data), (6, 1, 14, 14)).to(device) # The input data, in shape [6, 1, 14, 14]

          sample_size = 6

          # We can now run the prediction
          pred = model(model_input_data) # In shape [6, 10]

          # Answer, we will use y[0] though
          y = y.to(device)

          # Now, we will find the average array to multiply pred by
          # This array will be in shape [6, 1]
          mult = []
          t = 0
          time_decay = 18 # tau
          prev_x, prev_y = 0,0
          for i in range(sample_size):
            sample = arr[i]
            if t == 0: 
              # This is the edge case, dt will be 0 so the entire term will have no effect
              prev_x = sample.x1
              prev_y = sample.y1

            dt = math.sqrt(((prev_x - sample.x1)**2) + ((prev_y - sample.y1)**2)) / 2800 # nu is 1/2800
            prev_x, prev_y = sample.x1, sample.y1

            temp = []
            temp.append(max(0, (1/3) * (math.log((-t/time_decay) + 1, 1.5) + 1) * ((1 + dt) ** (1 + dt))))
            mult.append(temp) # This line is to make sure the array is in the write shape

            t += 1

          multT = torch.tensor(mult).to(device)
          pred = pred * multT
          # This averages the outputs
          actual_pred = torch.mean(pred, dim=0, keepdim=True, dtype=torch.float) # Shape is in [1, 10]
          test_loss += loss_fn(actual_pred, y).item()
          correct += (actual_pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"{100*correct:>0.3f}%")
