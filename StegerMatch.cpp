/**
 * StegerMatch.cpp
 * Hisham Bedri
 * November 2, 2015
 *
 * This implements the method from: C. Steger. Occlusion Clutter, and Illumination Invariant Object Recognition. In ISPRS, 2002.
 * Complete:
 * -Locates a tempalte within an image by searching over a parameter space of scale and rotation of the template.
 * -Templates are matched with the image in the gradient domain (dot product of X,Y gradients)
 * -Tested with three shifts of the template
 * -Tested with cluttered image
 * Todo:
 * -Form a pyramid for faster searching, once match is made on top level of the pyramid, traverse down (and over more parameters) to refine
 * -Try other normalizations for the similarity evaluation
 * -implement short-circuit for evaluation if template is not greater than running maximum
 */

#include <iostream>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class StegerMatch {
public:
    StegerMatch():mTemplateLoaded(false) {}
    
    /**
     * This method matches the pre-loaded template to 'input' and stores the per-pixel similary scores in 'output' 
     */
    void processFrame(const Mat& input, Mat& output) {
        if (!mTemplateLoaded) {
            cerr << "Template was not loaded!" << endl;
            return;
        }

		output.create(input.size(), CV_32FC1); //holder for output
		Mat input_gray = Mat::zeros(input.rows, input.cols, CV_32FC1); //holder for grayscale test image
		cvtColor(input, input_gray, cv::COLOR_BGR2GRAY); //convert to grayscale
		//GaussianBlur(input_gray, input_gray, Size(7, 7), 0, 0, BORDER_DEFAULT); //optional blur to remove noise

		//create derivative in x and y directions
		//hint: image derivatives dx and dy can be obtained easily with cv::Sobel(...) or cv::Scharr(...)

		Mat gradient_x = Mat::zeros(input.rows, input.cols, CV_32FC1);
		Mat gradient_y = Mat::zeros(input.rows, input.cols, CV_32FC1);

		cv::Sobel(input_gray, gradient_x, CV_32FC1, 1, 0, 3);
		cv::Sobel(input_gray, gradient_y, CV_32FC1, 0, 1, 3);

		Mat temp = Mat::zeros(output.rows, output.cols, CV_32FC1); //holder for dot product
		Mat temp_x = Mat::zeros(output.rows, output.cols, CV_32FC1); //holder for x part of dot product
		Mat temp_y = Mat::zeros(output.rows, output.cols, CV_32FC1); //holder for y part of dot product
		//for each in template set
		for (int i = 0; i < mTemplateSet.size(); i++){
			//convolve with template
			Mat kernel = mTemplateSet.at(i).gradient_image_x; //x derivative kernel
			kernel = kernel - mean(kernel); kernel = kernel / norm(kernel, NORM_L2, noArray()); //remove mean and normalize

			filter2D(gradient_x, temp_x, CV_32FC1, kernel);

			kernel = mTemplateSet.at(i).gradient_image_y; //y derivative kernel
			kernel = kernel - mean(kernel); kernel = kernel / norm(kernel, NORM_L2, noArray());
			filter2D(gradient_y, temp_y, CV_32FC1, kernel);

			temp = temp_x + temp_y; //total dot product
			
			//perform max with output and result
			output = max(temp, output);
			//todo: save orientation and scale which produced maximum at each pixel

			//todo: 
			std::cout << "Completed " << i << " of " << mTemplateSet.size() << std::endl;

		}

		double mini, maxi;
		minMaxLoc(output, &mini, &maxi);
		output = output - mini;	output = output / (maxi - mini); //normalize output between 0 and 1
        
    }
    
    /**
     * This method takes in a color image (CV_8UC3, RGB) and primes the class to do matching against it.
     */
    void loadTemplate(const Mat& templ) {
        
		Mat warp_mat(2, 3, CV_32FC1); //matrix to hold affine transformation

		//create template set holder, templates are defined in structs and saved in the vector: mTemplateSet
		for (int rr = 0; rr < mParameters.mNumRotationSteps; rr++){
			for (int ss = 0; ss < mParameters.mNumScaleSteps; ss++){

				float rotation = mParameters.mMinRotation + (mParameters.mMaxRotation - mParameters.mMinRotation) / mParameters.mNumRotationSteps * rr;
				float scale = mParameters.mMinScale + (mParameters.mMaxScale - mParameters.mMinScale) / mParameters.mNumScaleSteps * ss;;
				mTemplateStruct temp;
				temp.scale = scale;
				temp.rotation = rotation;
				
				//resize
				Mat templ_resized = Mat::zeros(templ.rows*scale, templ.cols*scale, templ.type());
				resize(templ, templ_resized, Size(templ_resized.cols, templ_resized.rows), 0, 0);

				//convert to grayscale
				Mat original_image = Mat::zeros(templ_resized.rows, templ_resized.cols, CV_32FC1);
				cvtColor(templ_resized, original_image, cv::COLOR_BGR2GRAY);
				//GaussianBlur(original_image, original_image, Size(7, 7), 0, 0, BORDER_DEFAULT);

				//derivative
				Mat processed_image_x = Mat::zeros(original_image.rows, original_image.cols, CV_32FC1);
				Mat processed_image_y = Mat::zeros(original_image.rows, original_image.cols, CV_32FC1);
				cv::Sobel(original_image, processed_image_x, CV_32FC1, 1, 0, 3);
				cv::Sobel(original_image, processed_image_y, CV_32FC1, 0, 1, 3);

				//rotate
				warp_mat = getRotationMatrix2D(Point2f(original_image.cols / 2, original_image.rows / 2), rotation, 1); //get rotation matrix with scale of 1
				warpAffine(processed_image_x, processed_image_x, warp_mat, processed_image_x.size()); // apply affine transformation
				warpAffine(processed_image_y, processed_image_y, warp_mat, processed_image_y.size()); // apply affine transformation

				temp.original_image = original_image;
				temp.gradient_image_x = processed_image_x;
				temp.gradient_image_y = processed_image_y;

				mTemplateSet.push_back(temp);

			}
		}

		mTemplateLoaded = true;
    }
    
    bool isTemplateLoaded() { return mTemplateLoaded; }

	float getMinScale(){ return mParameters.mMinScale; }
	
	float getMaxScale(){ return mParameters.mMaxScale; }

	int getNumScaleSteps(){ return mParameters.mNumScaleSteps;  }

	void setMinScale(float minScale){
		if (minScale > 0){
			mParameters.mMinScale = minScale;
		}
	}

	void setMaxScale(float maxScale){
		if (maxScale > 0){
			mParameters.mMaxScale = maxScale;
		}
	}

	void setNumScaleSteps(int numScaleSteps){
		if (numScaleSteps > 0){
			mParameters.mNumScaleSteps = numScaleSteps;
		}
	}


	float getMinRotation(){ return mParameters.mMinRotation; }

	float getMaxRotation(){ return mParameters.mMaxRotation; }

	int getNumRotationSteps(){ return mParameters.mNumRotationSteps; }

	void setMinRotation(float minRotation){
		if (minRotation > 0){
			mParameters.mMinRotation = minRotation;
		}
	}

	void setMaxRotation(float maxRotation){
		if (maxRotation > 0){
			mParameters.mMaxRotation = maxRotation;
		}
	}

	void setNumRotationSteps(int numRotationSteps){
		if (numRotationSteps > 0){
			mParameters.mNumRotationSteps = numRotationSteps;
		}
	}


private:
    bool mTemplateLoaded;
	struct mTemplateStruct{
		float scale;
		float rotation;
		cv::Mat gradient_image_x;
		cv::Mat gradient_image_y;
		cv::Mat original_image;
	};
	struct mParameterStruct{
		float mMinRotation=0;
		float mMaxRotation=360;
		float mNumRotationSteps = 10;
		float mMinScale=.1;
		float mMaxScale=2;
		float mNumScaleSteps=10;
	};
	mParameterStruct mParameters;
	std::vector<mTemplateStruct> mTemplateSet;
};


int main(int argc, char** argv) {
    cout << "Hello Steger2002 matcher!" << endl;
    
    Mat myTemplate = imread("C:/Users/hisham/stegerMatch/template.jpg");
    //create ROI on the template image to only include the object... fill in the right values
	myTemplate = myTemplate(Rect(Point(300, 150), Size(350, 200)));
    
    Mat myInput = imread("C:/Users/hisham/stegerMatch/input.jpg");
	//Mat myInput = imread("C:/Users/hisham/stegerMatch/template2.jpg");
    


    StegerMatch match;	
    match.loadTemplate(myTemplate);
	
    if (match.isTemplateLoaded()) {
        Mat myOutput;
        match.processFrame(myInput, myOutput);
        
		Mat saving_image=myOutput*256;
		saving_image.convertTo(saving_image, CV_8UC1);
		imwrite("output.jpg", saving_image);


        imshow("template",  myTemplate);
        imshow("input",     myInput);
        imshow("output",    myOutput);
        waitKey(0);
    } else {
        cerr << "Template was not loaded." << endl;
    }
	
	
    
    return 0;
}
