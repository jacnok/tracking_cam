class Person{
  
PImage picture;
PVector pos,avg=new PVector(0,0);
color head,shirt;
int Rectid,id=0,coloredpixels=0;
int colorthreshold=150;//*determins how far apart colors can be.bigger number is farther colors*//
Rectangle headRect,rect2;
float countdown=0,deletetime=800;
boolean search=false,unsure=false,expieried=false,selected=false;
float avgx;
float avgy;
float sumx=0;
float sumy=0;
float count=0;


 Person(Rectangle face,int rectid,int idOfPerson){
headRect=face;
head=colorAvg(face);
//rect2=new Rectangle(int(face.x),int(face.y+(face.height*1.5)),face.width,face.height/2);
//shirt=colorAvg(rect2);
Rectid=rectid;
id=idOfPerson;
avgx=headRect.x+headRect.width/2;
avgy=headRect.height+headRect.height/2;
pos=new PVector(headRect.x+headRect.width/2,headRect.height+headRect.height/2);
takePicture();

}
void takePicture(){
  //println("ran");
picture=createImage(int(headRect.width*2),int(headRect.height*1.5),RGB);
picture.loadPixels();
int loc,x1=0,y1=0,sx=int(headRect.x-headRect.width),sy=int(headRect.y-headRect.height/3),max=video.width*video.height;
  for(int i=0; i<picture.pixels.length;i++){
  loc=(sx+x1)+(sy+y1)*video.width;
    if(loc>0 &&loc <max){
    picture.pixels[i] =video.pixels[loc];
    }else{
    picture.pixels[i] =color(0);
     }
  x1++;
    if(x1>=headRect.width+headRect.width){
    x1=0;
    y1+=1;
     }
  }
picture.updatePixels();
}

void Update(){
  strokeWeight(3);
  if(!search){
  stroke(0,255,0);
  pos.x=headRect.x+headRect.width/2;
  pos.y=headRect.y+headRect.height/2;
  }else{
    if(unsure){
      //println(millis()-countdown);
      if((millis()-countdown)>deletetime){
      expieried=true;
      }
      Rectid=100;
      stroke(255,0,0);}else{
    stroke(head);}
}
fill(0,0);

rect(headRect.x,headRect.y,headRect.width,headRect.height);
//rect(rect2.x,rect2.y,rect2.width,rect2.height);

//if(search){expieried=true;}
if (search){
  stroke(0,0,255);
  int ey=headRect.y+headRect.height;
  int ex=headRect.x+headRect.width;
  int count=0;
  avg.set(0,0);
for(int y=headRect.y;y<ey;y++){
    for(int x=headRect.x;x<ex;x++){
      if(x<video.width &&y<video.height&&y>0&&x>0){
     int loc=x+y*video.width;
     try {
     color currColor=video.pixels[loc]; 
    float d=colordistsq(currColor,head);
      if(d-brightness(currColor)*10<colorthreshold){
        stroke(abs(255*((d/colorthreshold)-1)));
       for(float i=abs(10*((d/colorthreshold)-1));i<10;i++){
        avg.add(x,y);
        count++;
        
      }
      point(x,y);
      }
  } catch (ArrayIndexOutOfBoundsException e) {
    println(x+", "+y);
    //println(headRect.x+","+headRect.y+","+headRect.width+", "+headRect.height);
  }
      }
      
    }
  }
  if(count<colorthreshold/10){
    if(unsure==false){
  unsure=true;
  //println("help");
countdown=millis();}
}
  else if (count>2.5*colorthreshold && count<coloredpixels){unsure=false;}
    if(count>0){
      //println(count);
avg.div(count);
pos.add(avg);
pos.div(2);


    }
}
headRect=new Rectangle(int(pos.x-(headRect.width/2)),int(pos.y-(headRect.height/2)),headRect.width,headRect.height);
 textAlign(CENTER);
    textSize(64);
    fill(0);
    text(id, pos.x,pos.y);
 //println(headRect.x+","+headRect.y+","+headRect.width+", "+headRect.height);
}



float colordistsq(color c1,color c2){
  float r1=red(c1);
      float g1=green(c1); 
      float b1=blue(c1);
      float r2=red(c2);
      float g2=green(c2); 
      float b2=blue(c2);
  float d=(sq(r1-r2)+sq(g1-g2)+sq(b1-b2));
  return d;
}


void showface(int x,int y,int widthf){
picture.loadPixels();
picture.resize(video.width/widthf, 300);
  if (picture!=null){
  image(picture,x,y);
  }
}
color colorAvg(Rectangle R){
  int ex=R.x+R.width/2;
  int ey=R.y+R.height/2;
//  int r=0;
//  int g=0;
//  int b=0;
//  int count=0;
  int loc;
  color currColor;
//  for(int x=R.x;x<ex;x++){
//  for(int y=R.y;y<ey;y++){
//    if(x<video.width&&y<video.height){
  //loc=x+y*video.width;
  loc=ex+ey*video.width;
    currColor=video.pixels[loc];
//    if(brightness(currColor)>1){//determins how dark is to dark
//    r+=red(currColor);
//    g+=green(currColor);
//    b+=blue(currColor);
//    count++;
//    }
//    }}
//  }
//  if(count>0){
//    if (brightness(head)>20){//sets minimum brightness for color of head. no need to change
//  currColor=color(((r/count)+red(head))/2,((g/count)+green(head))/2,((b/count)+blue(head))/2);
//    }else{currColor=color(r/count,g/count,b/count);}
//  //currColor=(currColor+head)/2;
//  coloredpixels=count;
//} else{
//currColor=color(255,255,255);
//println("error at collorsearch");
//}
  return currColor;

}

Boolean facecheck(Rectangle R,int i){// This will only detect if face is in box
//if(pos.y<R.y+R.height){
  int range=70;
    if(pos.x>R.x-range&&pos.x<(R.x+R.width+range)){
      //add is color check
      
      Rectid=i;
      search=false;
      return true;
    }
//} 

//print(pos.y+"< "+(R.y+R.height)+", "+pos.x+" >"+(R.x+pos.x)+"< "+(R.x+R.width));
return false;
}
void getcolor(Rectangle R){
head=colorAvg(R);

}
//Boolean handcheck(Rectangle R,int i){// This will only detect if face is in box
//if(pos.y<R.y+R.height){
//    if(pos.x>R.x&&pos.x<(R.x+R.width)){
//      //add is color check
      
//      Rectid=i;
//      search=false;
//      return true;
//    }
//} 
//return false;
//}
}
