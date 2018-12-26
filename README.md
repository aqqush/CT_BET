# CT_BET: Robust Brain Extraction Tool for CT Head Images
To be able to run this you will need to have a GPU card available and have tensorflow and keras libraries installed in python.

In order to train the model on CT head images
1) clone the repository to your local drive. 
2) Put your images under 'image_folder' and masks under 'mask_folder'
3) run unet_CT_SS.py

In order to extract brain from a CT head image.
1) clone the repository to your local drive.
2) put your images under image_folder
3) set predict flag to be true in unet_CT_SS.py file
4) make sure you set "testLabelFlag" to be False, in the unet_CT_SS.py
5) run unet_CT_SS.py

Additionals:
==========================================
1)If you want to run the model on a new data without mask, you should set "testLabelFlag=False", which computes the DICE metric if you have masks in the mask_data folder. If you run metrics "testLabelFlag=True" on your new data with mask, make sure that you have them both with the same name in their folders.

2)You should also download the model weights into the weights_folder as instructed in text file within the weights_folder.

==========================================

Please contact me if you have any difficulty in running this code.

Email: akkus.zeynettin@mayo.edu

Zeynettin Akkus

Please cite the paper below if you use the tool in your study:
1) Zeynettin Akkus, Petro M. Kostandy, Kenneth A. Philbrick, Bradley J. Erickson. Proceedings Volume 10574, Medical Imaging 2018: Image Processing; 1057420 (2018) https://doi.org/10.1117/12.2293423

2) Zeynettin Akkus, Petro M. Kostandy, Kenneth A. Philbrick, Bradley J. Erickson. Robust Brain Extraction Tool for CT Head Images. In Press. Neurocomputing.
