# Text2App
Description: *create an app with 2 buttons. when button1 pressed, say "Happy Text to App!". when button2 pressed, play video starwars.mp4 .*


[![Text2App Live Creation of an app](http://img.youtube.com/vi/JtETeCqWX2U/0.jpg)](http://www.youtube.com/watch?v=JtETeCqWX2U "Text2App")


Create a functional Android app from a given text description of the app. This model generates a MIT app inventor export file that should be uploaded to MIT app inventor website to build the app. 
The current prototype supports 6 functionalities: 

1. Detect button press (Upto 4 buttons)
2. Detecting acceleration
3. Playing video
4. Playing music
5. Opening camera
6. Text2speech

A user can describe an app functionality in Natural Language with these 6 functionalities, and the model will create an MIT app inventor export file which can be uploaded to MIT app inventor cloud and be made into a functioning app.

![Text2App Generation Pipeline](https://raw.githubusercontent.com/Masum06/Text2App/master/text2app_diagram.jpg)
**Text2App Generation Pipeline**

## How to Upload to MIT App Inventor cloud:

* Visit http://ai2.appinventor.mit.edu/ and login with your Google Account.
* Click *My Project* > *Import Project .aia from my computer*
![Uploading to MIT App Inventor](https://raw.githubusercontent.com/Masum06/Text2App/master/app_inventor_upload.png)
* Select the *.aia* file generated from **Text2App** 
* Download *AI Companion* app for your Android Phone
* Click *Connect* > *AI Conpanion* > Scan the QR Code with the MIT AI2 Companion app in your phone
* Voila! You can now use the app

(c) Masum Hasan, 2020

Original project idea by: Dr. Zhijia Zhao
