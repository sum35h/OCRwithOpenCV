/*
ayoungprogrammer.blogspot.com

Part 1: Extracting contours from text

*/

#include <iostream>






#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/ml.h"
#include<stdlib.h>
#include <ctype.h>


using namespace std;
using namespace cv;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;
struct contour_sorter // 'less' for contours
{
    bool operator ()( const vector<Point>& a, const vector<Point> & b )
    {
        Rect ra(boundingRect(a));
        Rect rb(boundingRect(b));
        // scale factor for y should be larger than img.width
        return ( (50*ra.x + 1000*ra.y) <= (50*rb.x + 1000*rb.y) );
    }
};

class comparator{
public:
        bool operator()(vector<Point> c1,vector<Point>c2){

                return boundingRect( Mat(c1)).x<boundingRect( Mat(c2)).x;

        }

};

 cv::Mat matClassificationInts;      // these are our training classifications, note we will have to perform some conversions before writing to file later

                                // these are our training images, due to the data types that the KNN object KNearest requires, we have to declare a single Mat,
                                // then append to it as though it's a vector, also we will have to perform some conversions before writing to file later
    cv::Mat matTrainingImagesAsFlattenedFloats;




    void CreateXmlData()
    {
        std::cout << "training complete\n\n";

                // save classifications to file ///////////////////////////////////////////////////////

    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);           // open the classifications file

    if (fsClassifications.isOpened() == false) {                                                        // if the file was not opened successfully
        std::cout << "error, unable to open training classifications file, exiting program\n\n";        // show error message
        return;                                                                                      // and exit program
    }

    fsClassifications << "classifications" << matClassificationInts;        // write classifications into classifications section of classifications file
    fsClassifications.release();                                            // close the classifications file

                // save training images to file ///////////////////////////////////////////////////////

    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);         // open the training images file

    if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
        std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
        return;                                                                              // and exit program
    }

    fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;         // write training images into images section of images file
    fsTrainingImages.release();                                                 // close the training images file


    }



void extractContours(Mat& image,vector< vector<Point> > contours_poly){

    cv::Mat blank = cv::imread("b.jpg", 0);
    int no=0;



        //Sort contorus by x value going from left to right
        sort(contours_poly.begin(),contours_poly.end(),contour_sorter() );
        sort(contours_poly.begin(),contours_poly.end(),contour_sorter());


        //Loop through all contours to extract
         for( int i = 0; i< contours_poly.size(); i++ ){

                Rect r = boundingRect( Mat(contours_poly[i]) );


                Mat mask = Mat::zeros(image.size(), CV_8UC1);
                //Draw mask onto image
                drawContours(mask, contours_poly, i, Scalar(255), CV_FILLED);

               imshow("mask",mask);
                //Copy
                 Mat extractPic;
                 //Extract the character using the mask
                 image.copyTo(extractPic,mask);

                 Mat resizedPic = extractPic(r);

               /////////////////////////RESIZING//////& INVERSION //////////////////////////
               cv::resize(resizedPic, resizedPic, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
              // cv::bitwise_not(resizedPic,resizedPic);
               ///////////////////////////////////////////////////

                cv::Mat image=resizedPic.clone();

                //Show image
                imshow("image",image);
                //char ch  = waitKey(0);




   int intChar = cv::waitKey(0);

           if (intChar == 27) {        // if esc key was pressed
                return;              // exit program
            } else  {  //all caps letters A,B..Z

                matClassificationInts.push_back(intChar);       // append classification char to integer list of chars

                cv::Mat matImageFloat;                          // now add the training image (some conversion is necessary first) . . .
                resizedPic.convertTo(matImageFloat, CV_32FC1);       // convert Mat to float

                cv::Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);       // flatten

                matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);       // add to Mat as though it was a vector, this is necessary due to the
                   }                                                                         // data types that KNearest.train accepts
                /////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////////////////////////////


                stringstream searchMask;
                searchMask<<"F:/CWorkspace/OCR_Learning_part/output/";

              // std::cout<<"D="<<contours_poly[i+1].x-contours_poly[i];
                searchMask<<no++<<".jpg";
                imwrite(searchMask.str(),resizedPic);
                ////////////////////////////////////////////////////////////////////////////////
                ////////////////////////LEARNING////////////////////////////////////////
                     // get key press

                //////checking for blank spaces//////
                   if(i+1<contours_poly.size())
               {
                    stringstream str;
                 str<<"F:/CWorkspace/OCR_Learning_part/output/";
                    str<<no++<<".jpg";


                Rect rec1=boundingRect(contours_poly[i]);
                Rect rec2=boundingRect(contours_poly[i+1]);
                if(rec1.x+rec1.width+15
                   <rec2.x||(rec1.x>rec2.x&&rec1.y<rec2.y))
                   // cout<<"b "<<endl;
                   imwrite(str.str(),blank);


                //std::cout<<"X="<<boundingRect( Mat(contours_poly[i])).x<<endl;
           //cout<<"rec1 "<<rec1.x<<" width="<<rec1.width<<"r+w="<<rec1.x+rec1.width<<" r2="<<rec2.x<<endl;
           //cout<<"y="<<rec1.y<<endl;
             }

         }//EO for loop

std::cout << "training complete\n\n";
CreateXmlData();



}



void getContours(const char* filename)
{
  cv::Mat img = cv::imread(filename, 0);


  //Apply blur to smooth edges and use adapative thresholding

   cv::Size size(3,3);//size for blur

  cv::GaussianBlur(img,img,size,0);//kernel=[some mat stuff]*1/width*height

   adaptiveThreshold(img, img,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,75,10);
   //////////////////in,  out,max val, type of thresh, binary image, neighbours,some no to sub from mean that is calculated
   ////mean is calc bassed on the ponts of the neighbouthood...
   //result is a bin img

  cv::bitwise_not(img, img);//inverting the img...





  cv::Mat img2 = img.clone();

  //Trying to rotate the image to proper angle
  std::vector<cv::Point> points;

  cv::Mat_<uchar>::iterator it = img.begin<uchar>();// iterator is some type of ptr
  cv::Mat_<uchar>::iterator end = img.end<uchar>();
  for (; it != end; ++it)
    if (*it)//if the pixel is white  , prolly white is 1 and black is 0....
      points.push_back(it.pos());

  cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));//a rotated rect with min area wrt the bounding points,ie the text area

   double angle = box.angle;//inclined angle

  if (angle < -45.)
    angle += 90.; //makes the angle positive , so its in the first quad

  cv::Point2f vertices[4];
  box.points(vertices);
  for(int i = 0; i < 4; ++i)
    cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 1, CV_AA);//draws line



   cv::Mat rot_mat = cv::getRotationMatrix2D(box.center, angle, 1);//converts RotatedRect to Mat

   cv::Mat rotated;

  cv::warpAffine(img2, rotated, rot_mat, img.size(), cv::INTER_CUBIC);



  cv::Size box_size = box.size;
  if (box.angle < -45.)
    std::swap(box_size.width, box_size.height);
  cv::Mat cropped;

  cv::getRectSubPix(rotated, box_size, box.center, cropped);
  cv::imshow("Cropped", cropped);
  imwrite("rotatedOutput.jpg",cropped);

        Mat cropped2=cropped.clone();
cvtColor(cropped2,cropped2,CV_GRAY2RGB);

Mat cropped3 = cropped.clone();
cvtColor(cropped3,cropped3,CV_GRAY2RGB);

 vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Find contours
  cv:: findContours( cropped, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0) );



  /// Approximate contours to polygons + get bounding rects and circles
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  vector<Point2f>center( contours.size() );
  vector<float>radius( contours.size() );


  //Get poly contours
        for( int i = 0; i < contours.size(); i++ )
     {
                 approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
     }


  //Get only important contours, merge contours that are within another
  vector<vector<Point> > validContours;
        for (int i=0;i<contours_poly.size();i++){

                Rect r = boundingRect(Mat(contours_poly[i]));
                if(r.area()<100)continue;
                bool inside = false;
                for(int j=0;j<contours_poly.size();j++){
                        if(j==i)continue;

                        Rect r2 = boundingRect(Mat(contours_poly[j]));
                        if(r2.area()<100||r2.area()<r.area())continue;
                        if(r.x>r2.x&&r.x+r.width<r2.x+r2.width&&
                                r.y>r2.y&&r.y+r.height<r2.y+r2.height){

                                inside = true;
                        }
                }
                if(inside)continue;
                validContours.push_back(contours_poly[i]);
        }


        //Get bounding rects
        for(int i=0;i<validContours.size();i++){
                boundRect[i] = boundingRect( Mat(validContours[i]) );
        }


        //Display
  Scalar color = Scalar(0,255,0);
  for( int i = 0; i< validContours.size(); i++ )
     {
        if(boundRect[i].area()<100)continue;
      drawContours( cropped2, validContours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       rectangle( cropped2, boundRect[i].tl(), boundRect[i].br(),color, 2, 8, 0 );
     }

  imwrite("op.jpg",cropped2);
  imshow("Contours",cropped2);

  extractContours(cropped3,validContours);

cv::waitKey(0);

}





int main(void){

char fileName[256];
cin>>fileName;
getContours(fileName);

}
