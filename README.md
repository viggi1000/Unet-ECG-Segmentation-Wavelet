# Unet-ECG-Segmentation
This network uses Unet to perform segmentation of ECG to identify P,QRS,T components of a given ECG. 
# Annotation (or segmentation) of the electrocardiogram (ECG) with a long short-term memory neural network. 
Here, I experimented with performing segmentation of ECG a Unet architecture, The ECG from the QTDB dataset is converted to the wavelet domain using an edited version of the [PyTorchWavelets]https://github.com/tomrunia/PyTorchWavelets).
I started of by providing the ECG and my labels as inputs for applying wavelet transform and I store the corresponding scales and the real & imaginary parts of the wavelet.  
In the beginning I struggled a bit to get the input and output to match using the standard 1d Conv and 1d ConvTranspose, I labelled the P segment as 1 QRS as 2 and T segments as 3 using the the WFDB package.
<br>It appears to work well on the QT database of physionet, but there is some issue with the way the labelling is done; I have to look for other datasets to try.
 
## Model

```

```

## Getting Started
- Download QTDB ECG dataset using something <code>wget -r -l1 --no-parent https://physionet.org/physiobank/database/qtdb/</code> to qtdb directory
- Run the QTDB_Wavelet_Extract python notebook which will store wavelet transform of labels and ECG to wavelet_dataset directory
- Now run train_unet.py 
- Visualize results with test_unet.py

## Output
Training took about 30 mins on a GTX 1080ti. 
![figure_12_test](https://user-images.githubusercontent.com/1295467/43062517-b9c546be-8e76-11e8-9bce-27665927888a.png)


## Dependencies
- PyTorch 0.3
- Numpy, Matplotlib 
- wfdb,tqdm 
