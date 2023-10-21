
import processing.video.*;//library to use video
import java.awt.Rectangle;//library to use rectangles
import gab.opencv.*;//library for opencv face detection
import processing.serial.*;//library to communicte to esp32 through comms


boolean scanface=false,esp=false,algerithm2=false;//wether it starts folloing a face
Serial myPort;//object for port
int oldface=0,u=0,d=0,L=0,r=0,reset=0,Senitivity=255,zoom=50;
//^varible to store faces, move camera up,down,left,right,reset camera position,how sensitive motor is,zoom level 48-no movement 0-zoom out 99-zoom in
PVector aveFlow;
OpenCV opencv;
//object for opencv
int timedelay=0,selected=0,widthface=6,boundsx=150,boundsy=150;
// varible for saving time before adds face, selected face it will follow,how many faces it can have at bottom of screen,the X and Y balance for weather motor should turn or not

Capture video; //object for video
//Movie video;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -these  marks allow me to quickly change between testeing recorded video vs live video

Rectangle[] faces;//array of rectangles that will be opencvs faces
ArrayList<Person> peaple = new ArrayList<Person>();//array of peaple /objects 
boolean run=true;// stops app from telling motor to move when a key is pressed

//IMPORTANT each created person(object) must a have a face from open cv initaily paired to it. If opencv face it will use my algerithim(much more ineffecient) to find face and switch back to opencv whenever possible

void setup(){ //the setup that runs once to setup the app 
size(640,660); //size of screen/app
 background(0); //the background will be black
String[] cameras = Capture.list(); //creats a list of cameras detected
printArray(cameras);//prints how many cameres there are
String[] coms=Serial.list();
printArray(Serial.list());
//video=new Capture(this,width,height-300,"HK 2M CAM #2");

video=new Capture(this,width,height-300,cameras[0], 50); //opens camera as the size of the app leaving some room for facces at the bottom
//video=new Movie(this,"multiscan.mp4");//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//video=new Movie(this,"testpaster2.mp4");// difrent testing videos


opencv = new OpenCV(this, 640,360);//opencv scans the screen
//opencv.loadCascade(OpenCV.CASCADE_PROFILEFACE);
opencv.loadCascade(OpenCV.CASCADE_FRONTALFACE);//difrent ways opencv can detect faces
//opencv.loadCascade(OpenCV.CASCADE_RIGHT_EAR);

video.start();//starts viewing live video
//video.play();//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-views prerecorded video

if (Serial.list().length!=0){//if there is no conection to esp/comms it wont run
for(int i=0;i<Serial.list().length-1;i++){
  println(coms[i]+","+"/dev/cu.usbserial-0001");
  if (coms[i].contains("/dev/cu.usbserial-0001")){
myPort = new Serial(this, "/dev/cu.usbserial-0001", 115200); //connet to comms
esp=true;

  }
}

  }
frameRate(30);
}

void captureEvent(Capture video){//needed to load next frame of live video
  video.read();
}
//void movieEvent(Movie m) {//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!neede to load next frame of prerecorded video
//  m.read();
//}

void draw(){// draw runs as many times as it can and is looped (pretty much like using while(true))
  image(video,0,0);//loads video and displays it at top left corner
   if (video.width == 0 || video.height == 0){//if there is a problem with video app wont run
    return;}
  opencv.loadImage(video);//sends frame to open cv
  opencv.calculateOpticalFlow();
aveFlow = opencv.getAverageFlow();
  //Rectangle[] faces = opencv.detect(2D,1,100,50,width/3);//opencv outouts rectangles where it thinks faces are
Rectangle[] faces = opencv.detect();
  
  strokeWeight(.5);//how big it draws lines/rectangles
  fill(225,50);//makes the rectangle have a haze over it
   for(Rectangle f:faces){ //puts rectangle over evry face it finds
   rect(f.x,f.y,f.width,f.height);//how it makes rectangles
   }
  if(oldface!=faces.length){ //if it detects change in the amount of faces
    if (peaple.isEmpty() && faces.length > 0) { //if there is no peaple already detected
    for(int i=0;i<faces.length;i++){//go through all faces
      peaple.add(new Person(faces[i],i,peaple.size()));//add peaple
      updateface();//update bottom screen 
      }
    }
  }

  else if(faces.length>peaple.size()&&(millis()-timedelay)>700){//if more faces detected for more than .7 seconds
      for(int i=0;i<faces.length;i++){//go through faces
        println(peaple.size());
        Boolean createface=true; //temperary true variable
          for(int p=0;p<peaple.size();p++){ //goes through exsiting peaple
            if(peaple.get(p).search){//if peaple are being searched for 
              
              if(peaple.get(p).facecheck(faces[i],i)){//checks if face is close enough if it is it gets connected to person object
              createface=false;
              }
            }
          }
          if(createface){//if face is not close enough creates a new person
          if(peaple.size()==0){
          peaple.add(new Person(faces[i],i,peaple.size()));
          println("only one person");
          }else{
           peaple.add(new Person(faces[i],i,peaple.size()-1));}//add peaple
      updateface();//udpadte bottom screen
          }
       }
    }
    else if(faces.length<peaple.size()){//if there are less faces than pepale detected it will find face that is not connected to opencv face
       Boolean[] search=new Boolean[peaple.size()];//create a list of boolean values that are the size of the peaple
        for (int s=0;s<search.length;s++){
              search[s]=true;//sets all search values to true
            }
      
      for(int i=0;i<faces.length;i++){
        for(int p=peaple.size()-1;p>0;p--){//scans backwards through all faces and does a check list with the search bools to find face that is not lost
              if(peaple.get(p).facecheck(faces[i],i)){
               search[p]=false; 
              }
            else{peaple.get(p).getcolor(faces[i]);}
            }
          }
        for(int s=0;s<search.length;s++){
        if(search[s]=true){
           
          peaple.get(s).search=true; }//finds that face that is not connected to opencv face and uses my search algorithim to find face
        }
        if(peaple.size()>1){
        for(int p1=0;p1<peaple.size();p1++){
         
          for(int p2=p1+1;p2<peaple.size();p2++){
            if(distBox(peaple.get(p1),peaple.get(p2))){//scans through all faces, finds duplects and deletes
            deleteFace(p2);
            }
        }
        }}
        
    }
    else if(faces.length==peaple.size()){
      for(int i=0;i<faces.length;i++){
          for(int p=0;p<peaple.size();p++){ //if opencv faces are = to peaple it will check for any faces that not paied and  pair them 
            if(peaple.get(p).search){
              peaple.get(p).facecheck(faces[i],i);
              }
          }
      }
    }


if(algerithm2){
  if(peaple.size()>0){
peaple.get(selected).search=true;}
}

for(Person p:peaple){
    if (!p.search){
      if (p.Rectid<faces.length){// for evrey face that is paired it will send opencvs faces to it 
        if(distsq(p.headRect,faces[p.Rectid],100)){
        p.headRect=faces[p.Rectid];}
      }
    }
    p.Update();//will update the person object
}
for (int i=0;i<peaple.size();i++)
{
if(peaple.get(i).expieried){//if any faces cant be found for a certain amount of time they will be delted
  deleteFace(i);
}
}
if(oldface!=faces.length){ //updates time counter for before it adds a new person
  timedelay=millis();
oldface=faces.length;
}


if(scanface){//if it is going to follow face
if(esp){ // if connected to esp through coms
  if(run){//if anonther key is not being pressed
    if(peaple.size()>0){ //if ther is peaple
      if(selected<peaple.size()){
        //print(selected);
      Person per=peaple.get(selected);
      if (!per.unsure){
    folowFace(per.pos);} //will follow selected person and send codes throug coms port
    else{u=0;d=0;L=0;r=0;}
    if(per.expieried){
      u=0;d=0;L=0;r=0;
      //println("noooooooooooooooooooooooo");
      }
  myPort.write(u+","+d+","+L+","+r+","+zoom+","+int(Senitivity)+","+reset+";");}
       println(u+","+d+","+L+","+r+","+zoom+","+int(Senitivity)+","+reset+";");
  //print("running");
    }else {if (selected>0){selected--;}}//if selected person doen not exsit move to other person
  if (peaple.size()==0)
{u=0;d=0;L=0;r=0;
myPort.write(u+","+d+","+L+","+r+","+zoom+","+int(Senitivity)+","+reset+";");
       println(u+","+d+","+L+","+r+","+zoom+","+int(Senitivity)+","+reset+";");
}
}
}}


textSize(20);
fill(255);
text("boundsx: "+boundsx,50,25);
text("boundsy: "+boundsy,50,50);
text("AI: "+scanface,50,75);
}


void mousePressed(){ //if the mosue is pressed it will Put a box around the selection Zone and delete any faces it finds
fill(225,100,100,60);
rect(mouseX-75,mouseY-75,150,150);  
Rectangle mr=new Rectangle(mouseX-75,mouseY-75,150,150);
for(int i=0;i<peaple.size();i++){
         Person p=peaple.get(i);
         if(distBox(p,mr)){
         deleteFace(i);
         }
  
  }



}


void deleteFace(int i){ //Will delete face update bottom of screen 
//if(selected==i){
//u=0;d=0;L=0;r=0;
//  myPort.write(u+","+d+","+L+","+r+","+zoom+","+int(Senitivity)+","+reset+";");
//       println(u+","+d+","+L+","+r+","+zoom+","+int(Senitivity)+","+reset+";");
//}
peaple.remove(i); // delet face
         noStroke();
   fill(0,255);
   rect(0,height-300,width,height);// Redraws bottom of the screen 
if(peaple.size()>0){
   int xface=0;
   for(Person p:peaple){//Searches through existing people and places their face at the bottom 
     p.showface(xface,height-300,widthface);
    xface+=(video.width/widthface)+20;
   }        
}
}



Boolean distBox(Person p,Person p2){ //Algorithm used to determine if person is overlaping another person
  PVector v; //Creates a  2d vector
  float dist;//saves Distance
  float bigdist=max(sq(p.headRect.width/2)+sq(p.headRect.height/2),sq(p2.headRect.width/2)+sq(p2.headRect.height/2));
 //^Determines the biggest that the distance can be to be within the Box
  v=p.pos.sub(p2.pos);
 dist=v.magSq();//Subtracts the distance between the people and finds the magnitude
 
  if (dist<bigdist){
    return true;
}//if in box
else {
return false;
}}
Boolean distBox(Person p,Rectangle r){// same as  distBox but for a rectangle instead
  if(p.avg.y>=r.y){//If the rectangle's Y position is less than the person that y position
  //^This eliminates it detecting the hand as the face
    if(p.avg.x>r.x&&p.avg.x<(r.x+r.width)){//If it is within the distance of the face returns true
    return true;
    }}
  return false;
}
void updateface(){ // Update bottom of the screen to have correct faces
if(peaple.size()>0){
   int xface=0;
   for(Person p:peaple){
     p.showface(xface,height-300,widthface);
    xface+=(video.width/widthface)+20;
   }        

if(scanface){//This draws the rectangle around the selected face
  strokeWeight(3);
stroke(0,255,0);
fill(0,0);
rect(video.width/widthface*selected,height-300,video.width/widthface*(selected+1),height);
}
}
}





void folowFace(PVector pos) {//This is for auto following face
 L=0;
 r=0;
 u=0;
 d=0;
Senitivity=0;
float checker=0;
float div=3;
float divy=3;
if(pos.x<boundsx){//Divides video into three sections and determines which sid the face is on
  checker=abs((((boundsx)-pos.x)/boundsx)*10); //Sets Checker equal to how far the face is from the side
  if (Senitivity<checker){
//Senitivity=int(checker);}
Senitivity=int(1);}
L=1;
strokeWeight(2);
line(boundsx,0,boundsx,video.height);//Puts a line in the direction it is turning
}
else if(pos.x>video.width-boundsx){//same thing as abouve but for right
  checker=abs((((video.width-boundsx)-(pos.x))/boundsx)*10);
  if (Senitivity<checker){
//Senitivity=int(checker);}
Senitivity=int(1);}//It will choose the highest sensitivity

r=1;
strokeWeight(2);
line(video.width-boundsx,0,video.width-boundsx,video.height);
}
if(pos.y<boundsy/2){//same thing as above but for up direction
  checker=abs(((pos.y-boundsy)/boundsy)*10);
  if (Senitivity<checker){
//Senitivity=int(checker);}
Senitivity=int(1);}
u=1;
strokeWeight(2);
line(0,boundsy/2,video.width,boundsy/2);
}else if(pos.y>video.height-(boundsy+boundsy/2)){
  checker=abs((((video.height-(boundsy+boundsy/2))-(pos.y))/boundsy)*10);
  if (Senitivity<checker){
//Senitivity=int(checker);}
Senitivity=int(1);}
d=1;
strokeWeight(2);
line(0,video.height-(boundsy+boundsy/2),video.width,video.height-(boundsy+boundsy/2));
}

}
///edit
void keyPressed(){ //These are controls for keyboard
  if(esp){
  run=false;//Disables Auto tracking while you are pressing a button
   u=0;d=0;L=0;r=0;Senitivity=1;
if (key=='a'){
L=1;
}else{
L=0;
}
if (key=='d'){
r=1;
}else{
r=0;
}
if (key=='w'){
u=1;
}else{
u=0;
}
if (key=='s'){
d=1;
}else{
d=0;
}
if (key=='r'){
reset=1;
}else{
reset=0;
}
if (key=='q'){
scanface=!scanface;
}
if (key=='o'){
  if(selected!=0){
selected--;}
}
if (key=='p'){
  if(selected<peaple.size()){
selected++;}
}
if (key=='z'){
  zoom++;
}
if (key=='x'){
  zoom--;
}
if (key=='e'){
algerithm2=!algerithm2;
print(algerithm2);
}

 myPort.write(u+","+d+","+L+","+r+","+zoom+","+int(Senitivity)+","+reset+";");
       println(u+","+d+","+L+","+r+","+zoom+","+int(Senitivity)+","+reset+";");
}
if (key=='g'){
  if(boundsx<(video.width/2)){
    boundsx+=25;  }
}
if (key=='h'){
   if(boundsx>0){
    boundsx-=25;}
}
if (key=='v'){
  if(boundsy<(video.height/2)){
    boundsy+=25; }
}
if (key=='b'){
    if(boundsy>0){
    boundsy-=25;}
}
if(key=='c'){
  if(peaple.size()>0){ 
for (int i=0;i<peaple.size();i++){
peaple.remove(i);
}}
}
stroke(0,255,0);
strokeWeight(5);
line(boundsx,0,boundsx,video.height-10);
line(video.width-boundsx,0,video.width-boundsx,video.height-10);
line(0,boundsy/2,video.width,boundsy/2);
line(0,video.height-(boundsy+boundsy/2),video.width,video.height-(boundsy+boundsy/2));
}
void keyReleased(){ //If Keys released sets all values to zero 
  if(esp){
  run=true;
  u=0;d=0;L=0;r=0;//ddzoom=50;
 myPort.write(u+","+d+","+L+","+r+","+zoom+","+int(Senitivity)+","+reset+";");
       println(u+","+d+","+L+","+r+","+zoom+","+int(Senitivity)+","+reset+";");
  }
}

Boolean distsq(Rectangle P, Rectangle F,float threshold){
  float x=((P.x-(P.width/2))-(F.x-(F.width/2)));
  float y=(P.y-(P.height/2))-(F.y-(F.height/2));
if((sq(x)+sq(y))<sq(threshold)){
return true;
}else {return false;}
}
