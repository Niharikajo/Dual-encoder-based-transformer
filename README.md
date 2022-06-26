# Dual-encoder-based-transformer
 ## :paperclip: Overview
 • :file_folder: [`Data/dataset`](https://github.com/Niharikajo/Dual-encoder-based-transformer/tree/main/data) contains the datasets used for the experiments. </br>
 • :page_facing_up: [`data_loader.py`](https://github.com/Niharikajo/Dual-encoder-based-transformer/blob/main/data/data_loader.py) is where most of the data  preprocessing happens. </br>
 • :file_folder: [`models`](https://github.com/Niharikajo/Dual-encoder-based-transformer/tree/main/models) holds the model architecture. </br>
 #
 
 **:round_pushpin:Code has been executed in Google Colaboratory.**
 #
 
 ## Code Execution on Google Colab 
• Download the code, unzip it and upload it to drive. </br>
• Mount the drive on google colab. </br>
• Install any missing dependencies. </br>
• Run the following python script </br>
`!python -u main_yformer.py --model yformer --data Battery --train_epochs 4 --attn prob --freq t --features S` </br>

--------------------------------------------------------------------------------------------------------------
## Denormalizing Test Data
Code:
```
outputs= outputs.detach().cpu().numpy()
outputs = outputs.reshape(-1,outputs.shape[-2])
outputs = self.Data.inverse_transform(outputs)

batch_y = batch_y.detach().cpu().numpy()
batch_y = batch_y.reshape(-1,batch_y.shape[-2])
#print('batch_y', batch_y.shape)
batch_y = self.Data.inverse_transform(batch_y)

pred = outputs
true = batch_y

preds.append(pred)
trues.append(true)
```

The above code is included in exp_informer </br>
Preds and trues are reshaped as follows before error metric calculation
```
preds = preds.reshape(-1, preds.shape[-1], 1)
trues = trues.reshape(-1, trues.shape[-1], 1)
```

-----------------------------------------------------------------------------------------------------------------
## Code Documentation
The following taable contains the list of parameters used :

| Parameters    | Values |
| ------------- | ------------- |
|seq_len (sequence_length)|48|
|label_len|48|
|enc_in (encoder input size)|1|
|dec_in (decoder input size) |1|
|c_out(output size)|1|
|d_model (dimension of model)|512|
|n_heads  (num of heads)|8|
|e_layers (num of encoder layers)|3|
|d_layers (num of decoder layers)|3|
|d_ff (dimension of fcn)|2048|
|Factor (probsparse attn factor)|3|
|dropout|0.05|
|Alpha | 0.7|
|learning_rate|0.0001|

#

The code execution begins from `main_yformer.py`. </br>
### 1. main_yformer.py

``` python
data_parser = {
    'ETTh1':{'data':'ETTh1.csv', 'T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]}, 
    'Battery': {'data':'dataset4.csv', 'T':'Voltage(V)', 'M':[2,2,2], 'S':[1,1,1], 'MS':[2,2,1]}, 
}
```

'Battery' is slected from data_parser </br> 

``` python 
if args.data in data_parser.keys():
    data_info = data_parser[args.data] # Battery is selected (data in args is battery)
    args.data_path = data_info['data'] # data_path = dataset4.csv
    args.target = data_info['T'] # target = Voltage(V)
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]  # enc_in =1,dec_in = 1,c_out = 1

args.detail_freq = args.freq # freq = t
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

```
The args are printed as follows: </br> 

 
Args in experiment: </br> 
Namespace(activation='gelu', alpha=0.7, attn='prob', batch_size=32, c_out=1, checkpoints='./checkpoints/', d_ff=2048, d_layers=1, d_model=512, data='Battery', data_path='dataset4.csv', dec_in=1, des='test', detail_freq='s', devices='0,1,2,3', distil=True, do_predict=False, dropout=0.05, e_layers=2, embed='timeF', enc_in=1, factor=5, features='S', freq='s', gpu=0, itr=1, label_len=48, learning_rate=0.0001, loss='mse', lradj='type1', model='yformer', n_heads=8, num_workers=0, output_attention=False, patience=3, pred_len=24, root_path='./data/ETT/', seq_len=48, target='Voltage(V)', train_epochs=4, use_amp=False, use_decoder_tokens=0, use_gpu=True, use_multi_gpu=False, weight_decay=0.0)

``` python
Exp = Exp_Informer 
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_tk{}_wd{}_lr{}_al{}_{}_{}'.format(args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.use_decoder_tokens, args.weight_decay, args.learning_rate, args.alpha, args.des, ii)
      exp = Exp(args) # object (exp) of class Exp_Informer is created
```
Exp_basic is the parent class of Exp_informer, so __innit__ of exp_basic is executed first.

 ### 2.	exp_basic.py
 ``` python
 class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
```
output: </br>
Use GPU: cuda:0

### 3. exp_informer.py
``` python
class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)   # https://realpython.com/python-super/
def _build_model(self):
        model_dict = {
            'informer':Informer,
            'yformer':Yformer,

        }
        if self.args.model=='informer'or self.args.model=="yformer":
            model = model_dict[self.args.model](                       #model is a object of class Yformer or Informer(whichever is selected)
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            # https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
```
 Back to main_yformer.py
 
 ``` python
 print('start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
```

output: </br>
start training : yformer_Battery_ftS_sl48_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_tk0_wd0.0_lr0.0001_al0.7_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>

Going back to exp_informer to train the model

### 3. exp_informer.py

``` python
def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
 ```
 _get_data function in exp_informer is called
 
 ``` python
 def _get_data(self, flag):
        args = self.args

        data_dict = {
            'Battery':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data] 
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        
        data_set = Data(                             # Data = Dataset_Custom
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            use_decoder_tokens=args.use_decoder_tokens,
            timeenc=timeenc,
            freq=freq
        )
        print(flag, len(data_set))
        #__len__(self) is automatically called when len(obj) is used 
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader
    # https://pytorch.org/docs/stable/optim.html
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim
    ##https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
```
The data split and other preprocessing is done in data_loader.py, Dataset_Custom class is initialized. </br>
All the dimensions/size mentioned in comment are for first iteration training of dataset4.

### 4. data_loader.py
``` python
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
features='S', data_path='dataset.csv', use_decoder_tokens=False,
target='Voltage(V)', scale=True, timeenc=1, freq='t', sample_frac=1):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.use_decoder_tokens = use_decoder_tokens
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,    #csv file is read
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]

        '''
        #dataset is rearranged in the format [date , other features, target feature ]
        ftrs = []
        for i in range((df_raw.shape[1] - 2)):
            ftrs.append('ftr' + str(i))  # ftrs = ['ftr0']
        header = list(['date'] + ftrs + [self.target])  #header = ['date', 'ftr0', 'Voltage(V)']
        df_raw.columns = header

        cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        #test train val split
        num_train = int(len(df_raw)*0.6818) #2815
        num_test = int(len(df_raw)*0.1818) #750
        num_vali = len(df_raw) - num_train - num_test #564
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len] #[0, 2767, 3331]
        border2s = [num_train, num_train+num_vali, len(df_raw)] #[2815, 3379, 4129]
        border1 = border1s[self.set_type] #0
        border2 = border2s[self.set_type] #2815
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values) #Compute the mean and std to be used for later scaling.
            data = self.scaler.transform(df_data.values) #Perform standardization by centering and scaling.
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2] # 0 to 2815 dates are selected
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc==0:
            df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
            df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
            data_stamp = df_stamp.drop(['date'],1).values
        elif self.timeenc==1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1,0)

        self.data_x = data[border1:border2] #0 to 2815 indexed target feature are selected
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.use_decoder_tokens:
            # decoder without tokens
            r_begin = s_end
            r_end = r_begin + self.pred_len

        else:
            # decoder with tokens
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]#example [150:198]
        seq_y = self.data_y[r_begin:r_end] #[198:222]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1 # 2815-48-24+1 = 2744

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
 ```
 
 #
 
 The preprocessing on the data is done as follw0s: </br>
Target feature(voltage) is converted into an array </br>
```
array([[3.39369583],
       [3.39438534],
       [3.39477348],
       ...,
       [3.97363758],
       [3.93240547],
       [3.89773369]])
```
Standard scalar is applied  on the above data </br>
```
array([[-1.26030578],
       [-1.25873246],
       [-1.25784679],
       ...,
       [ 1.03573542],
       [ 1.03588775],
       [ 1.03582464]]) 
```
Date is converted into datetimeindex </br>
```
DatetimeIndex(['2018-01-27 15:48:10', '2018-01-27 15:50:10',
               '2018-01-27 15:52:10', '2018-01-27 15:54:10',
               '2018-01-27 15:56:10', '2018-01-27 15:58:10',
               '2018-01-27 16:00:10', '2018-01-27 16:02:10',
               '2018-01-27 16:04:10', '2018-01-27 16:06:10',
               ...
               '2018-01-31 13:18:10', '2018-01-31 13:20:10',
               '2018-01-31 13:22:10', '2018-01-31 13:24:10',
               '2018-01-31 13:26:10', '2018-01-31 13:28:10',
               '2018-01-31 13:30:10', '2018-01-31 13:32:10',
               '2018-01-31 13:34:10', '2018-01-31 13:36:10'],
              dtype='datetime64[ns]', length=2815, freq=None)
```
Time feature encoding is done on Date, which is done in Time_fearture.py
#
### 5.Time_fearture.py
``` python
def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
```

``` python
def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str) #offset = second

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)
 ```
 ``` python     
 class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"
```
``` python
class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5
```

If frequency selected is second(s), Date is encoded into 6d array </br>
If frequency is minute(t), 5d array, </br>
If frequency is hour(h), 3d array. </br>
Considering frequency as minute(t), Date is encoded as follows

```
[[ 0.31355932  0.15217391  0.33333333  0.36666667 -0.42876712]
 [ 0.34745763  0.15217391  0.33333333  0.36666667 -0.42876712]
 [ 0.38135593  0.15217391  0.33333333  0.36666667 -0.42876712]
 ...
 [ 0.04237288  0.06521739 -0.16666667  0.5        -0.41780822]
 [ 0.07627119  0.06521739 -0.16666667  0.5        -0.41780822]
 [ 0.11016949  0.06521739 -0.16666667  0.5        -0.41780822]]
```
#
Returning to exp_informer.py
``` python
 # Generate a file path for storing checkpoints
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            #Automatic mixed precision model, Mixed precision tries to match each op to its appropriate datatype
            #https://pytorch.org/docs/stable/amp.html
            scaler = torch.cuda.amp.GradScaler() 
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
             # visualization of the model,  returns the complete architecture of the model
            summary(self.model,  [batch_x.shape, batch_x_mark.shape, batch_y.shape, batch_y_mark.shape]) # show the size 
            break
  ```
  _select_optimizer, _select_criterion is defined as follows:
  
  ``` python
  # https://pytorch.org/docs/stable/optim.html
def _select_optimizer(self): 
model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        	return model_optim
#https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
def _select_criterion(self):
 criterion =  nn.MSELoss()
        	  return criterion
```
``` python
 for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            auto_train_loss = []
            combined_train_loss = []
            self.model.train() #Call the model for training
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html               
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)#48(target feature)
                batch_y = batch_y.float() #next 24(target feature)
                
                batch_x_mark = batch_x_mark.float().to(self.device)#48(encoded timestamps)
                batch_y_mark = batch_y_mark.float().to(self.device) #24(encoded timestamps)

                # decoder input
                #Returns a tensor filled with the scalar value 0, with the same size as input.
                #returns a zero tensor of size 24 

                dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
                
                if self.args.use_decoder_tokens:
                    dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = dec_inp.float().to(self.device)
                
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
```
#

model initialization is done in exp_informer, build_model, i.e all objects are already created and __innit__ functions of the model is executed. </br>
Model execution starts from model.py </br>
### 6. model.py
``` python 
class Yformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True,
                device=torch.device('cuda:0')):
        super(Yformer, self).__init__()
        self.pred_len = out_len
        self.seq_len = seq_len
        self.attn = attn
        self.output_attention = output_attention
```
``` python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns, x_list = self.encoder(enc_out, attn_mask=enc_self_mask)
        x_list.reverse()
        # print("input shape x_dec, x_mark_dec",  x_dec.shape, x_mark_dec.shape)

        # Future Encoder
        fut_enc_out = self.fut_enc_embedding(x_dec, x_mark_dec)
        fut_enc_out, attns, fut_x_list = self.future_encoder(fut_enc_out, attn_mask=enc_self_mask)
        fut_x_list.reverse()

        # Decoder
        dec_out, attns = self.udecoder(x_list, fut_x_list, attn_mask=dec_self_mask)

        seq_len_dec_out = self.pred_len_projection(dec_out)[:, -(self.seq_len):,:]
        pre_len_dec_out = self.seq_len_projection(dec_out)[:, -(self.pred_len):,:]

        dec_out = torch.cat((seq_len_dec_out, pre_len_dec_out), dim=1)  # 336 -> 336 + 336
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out # [B, L, D]
```
 Before entering the encoder data must first be embeded
``` python
#Embedding
 self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout) #enc_in=1, d_model=512,
 self.fut_enc_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
```
### 7. embed.py 
Data embedding consists of Positional Embedding, Token Embedding, Temporal Embedding.
``` python 
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,               #https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu') #https://pytorch.org/docs/stable/nn.init.html

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x
```
``` python 
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
```
``` python 
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        #https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()
```
``` python
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'): #1,512
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x
```
``` python
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model) #1, 512
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        return self.dropout(x)
```
#
Next is the past encoder part
``` python
 # PastEncoder
        self.encoder = YformerEncoder(
            [
                # uses probSparse attention
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
```
### 8. encoder.py

``` python
class YformerEncoder(nn.Module):
    def __init__(self, attn_layers=None, conv_layers=None, norm_layer=None):
        super(YformerEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers) if attn_layers is not None else None
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        x_list = []
        x_list.append(x)
        if self.conv_layers is not None:
            # print("Conv layers not none")
            if self.attn_layers is not None:
                for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                    x, attn = attn_layer(x, attn_mask=attn_mask)
                    x = conv_layer(x)
                    x_list.append(x)
                    attns.append(attn)
                # x, attn = self.attn_layers[-1](x)
                # x_list.append(x)
                # attns.append(attn)
            else:
                # pipeline for only convolution layers
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)
                    x_list.append(x)
                    attns.append(None)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x_list.append(x)
                attns.append(attn)

        if self.norm is not None: 
            x = self.norm(x)
            for i in range(len(x_list)): # add this norm to every output of x_list
                x_list[i] = self.norm(x_list[i])
        return x, attns, x_list
```
``` python
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn
```
``` python 
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)

        return x
```
#
### 9. attn.py 
Attention Layers, Probsparse attention, Masked attention which are used in the encoder are defined as follows:
``` python
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads) #64
        d_values = d_values or (d_model//n_heads) #64
        #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads) 
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape #32,48
        _, S, _ = keys.shape #48
        H = self.n_heads #8

        queries = self.query_projection(queries).view(B, L, H, -1) #32,48,8,64
        keys = self.key_projection(keys).view(B, S, H, -1) #32,48,8,64
        values = self.value_projection(values).view(B, S, H, -1) #32,48,8,64

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
```
``` python
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
     def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape #32,48,8,8
        _, L_K, _, _ = keys.shape #48

        queries = queries.transpose(2,1) #32,8,48,64
        keys = keys.transpose(2,1) #32,8,48,64
        values = values.transpose(2,1) #32,8,48,64

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # 20
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # 20

        U_part = U_part if U_part<L_K else L_K #20
        u = u if u<L_Q else L_Q #20
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.contiguous(), attn
    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape #32,8,48,64
        _, _, L_Q, _ = Q.shape #32,8,48,64

        # calculate the sampled Q_K
        #https://pytorch.org/docs/stable/generated/torch.unsqueeze.html, Returns a new tensor with a dimension of size one inserted at the specified position
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E) #[32, 8, 1, 48, 64], #[32, 8, 48, 48, 64]
        ##https://pytorch.org/docs/stable/generated/torch.randint.html, Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # tensor of dimension(48,20) is cretaed filled with random values between 0 and 48
        
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :] #[32, 8, 48, 20, 64]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze() #matmul([32, 8, 48, 1, 64],[32, 8, 48, 64, 20])
        #[32, 8, 48, 20]
        #https://pytorch.org/docs/stable/generated/torch.max.html
        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K) #[32,8,48]
        #https://pytorch.org/docs/stable/generated/torch.topk.html
        #Returns the k largest elements of the given input tensor along a given dimension.
        M_top = M.topk(n_top, sorted=False)[1] #[32,48,20]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q) [32, 8, 20, 64]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top
     def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape #[32,8,48,64]
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            #https://pytorch.org/docs/stable/generated/torch.cumsum.html
            contex = V.cumsum(dim=-2)
        return contex
     def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape #32,48,8,64

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            #https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)
```
Probsparse attention uses ProbMask.
``` python
class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"): #32, 48, 48
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask
```
#
Next is the future encoder, which is defined as,
``` python
        # Future encoder
        self.future_encoder = YformerEncoder(
            [
                # uses masked attention
                EncoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
```
Encoder layer is used is similar to past encoder, but here were use Masked Attention(Full Attention) instead of probsparse attention.

``` python 
#masked attn
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
```
Here we used TriangularCausalMask
``` python
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
```
#
Next is the decoder block </br>
``` python
        # Decoder
        self.udecoder = YformerDecoder(
            [
                
                YformerDecoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(d_layers)
            ],
            [
                DeConvLayer(
                    d_model
                ) for l in range(d_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
```

### 10. decoder.py
``` python 
class YformerDecoder(nn.Module):
    def __init__(self, attn_layers=None, conv_layers=None, norm_layer=None):
        super(YformerDecoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers) if attn_layers is not None else None
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.first_conv = DeConvLayer(c_in=512)
        self.first_attn = YformerDecoderLayer(AttentionLayer(FullAttention(False), d_model=512, n_heads=8), d_model =512, d_ff = 2048)
        self.norm = norm_layer

    def forward(self, x_list, fut_x_list, attn_mask=None):
        # x [B, L, D]
        attns = []
        x = x_list.pop(0)
        fut_x = fut_x_list.pop(0)
        x = self.first_attn(x, fut_x, cross_mask=attn_mask)
        x = self.first_conv(x) # upsample to connect with other layers from encoder
        if self.conv_layers is not None:
            if self.attn_layers is not None:
                for cross_x, cross_fut_x, attn_layer, conv_layer in zip(x_list, fut_x_list, self.attn_layers, self.conv_layers):
                    cross = torch.cat((cross_x,cross_fut_x), dim=1)
                    x = attn_layer(x, cross, cross_mask=attn_mask)
                    x = conv_layer(x)
            else:
                # pipeline for only convolution layers
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None: 
            x = self.norm(x)
        return x, attns
```
``` python 
class YformerDecoderLayer(nn.Module):
    def __init__(self,  cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(YformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        # self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, cross_mask =None):

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        x = x + self.dropout(x)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)
```
``` python 
class DeConvLayer(nn.Module):
    def __init__(self, c_in, c_out =None):
        super(DeConvLayer, self).__init__()
        c_out = c_in if c_out is None else c_out
        self.upConv = nn.ConvTranspose1d(in_channels=c_in,
                                  out_channels=c_out,
                                  kernel_size=3,
                                  stride=2,
                                  padding=2)
        self.norm = nn.BatchNorm1d(c_out)
        self.activation = nn.ELU()
        # self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.upConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        # x = self.maxPool(x)
        x = x.transpose(1,2)
        return x
```
There are two attentions used, masked and probsparse attention, which are defined in the encoder section.
#
Next is the Linear Layer
``` python
self.seq_len_projection = nn.Linear(d_model, c_out, bias=True) 
self.pred_len_projection = nn.Linear(d_model, c_out, bias=True) 
```
#
The outputs are received in exp_informer, error metrices are calculated and next iteration continues.
``` python
                f_dim = -1 if self.args.features=='MS' else 0
                batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
                auto_loss = criterion(outputs[:, :-self.args.pred_len,:], batch_x.view(1, -1))
                auto_train_loss.append(auto_loss.item())
                loss = criterion(outputs[:, -self.args.pred_len:,:], batch_y.view(1, -1))
                train_loss.append(loss.item())

                combined_loss = self.args.alpha * auto_loss + (1-self.args.alpha) * loss
                combined_train_loss.append(combined_loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | auto loss: {3:.7f} | comb loss: {4:.7f}".format(i + 1, epoch + 1, loss.item(), auto_loss.item(), combined_loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                combined_loss.backward()
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            auto_loss = np.average(auto_train_loss)
            combined_loss = np.average(combined_train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Auto Loss : {3:.7f} | Comb Loss : {4:.7f}, Vali Loss: {5:.7f} Test Loss: {6:.7f}".format(
                epoch + 1, train_steps, train_loss, auto_loss, combined_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
```
#
After training is completed the model is tested with test data.
``` python
 print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
```
``` python
def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
            if self.args.use_decoder_tokens:
                dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            else:
                dec_inp = dec_inp.float().to(self.device)
            # encoder - decoder
            
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
            
            pred = outputs.detach().cpu().numpy()#.squeeze()
            true = batch_y.detach().cpu().numpy()#.squeeze()
            
            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return
```
#
### 11. metrics.py
``` python
def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe
```
---------------------------------------------------------------------------------------
## Acknowledgments
Dataset source : Diao, W., Saxena, S., Pecht, M. Accelerated Cycle Life Testing and Capacity Degradation Modeling of LiCoO2 -graphite Cells. J. Power Sources 2019, 435, 226830. https://web.calce.umd.edu/batteries/data.htm



