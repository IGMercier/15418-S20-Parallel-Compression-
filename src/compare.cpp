/*BY Ismael Mercier, imercier@andrew.cmu.edu
 * 04/24/20
 *This file is for the purposes of comparing compressed and
 *uncompressed images or those compressed by a sequential compressor
 *and those compressed by a parallel compressor
 */

#include <string>
#include <iostream>
#include <fstream>

#define TOLERANCE 1
/*
 *func: compares two images
 *param: the directory of each of the images
 *output: how many pixels don't match, -1 if something
 *        failed, 0 if perfect match
 */

int compare_count(string dir0, string dir1)
{
    //file extension
    string ext=".jpg";

    //open the two images
    Mat img0=imread(dir0+ext, CV_LOAD_IMAGE_GRAYSCALE);
    Mat img1=imread(dir1+ext, CV_LOAD_IMAGE_GRAYSCALE);

    //compare size
    n0 = img0.rows;
    m0 = img0.cols;
    n1 = img1.rows;
    m1 = im1.cols;
    if(n0 != n1 || m0 != m1)
    {
        cout << "image sizes did not match!\n"
        cout << "img0:("<<n0<<","<<m0<<")\n";
        cout << "img1:("<<n1<<","<<m1<<")\n";
        return -1;
    }

    //compare how many pixels are wrong
    int count = 0;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            Vec3b pixel0 = img0.at<Vec3b>(i,j);
            b0 = pixel0.val[0];
            g0 = pixel0.val[1];
            r0 = pixel0.val[2];
            Vec3b pixel1 = img1.at<Vec3b>(i,j);
            b1 = pixel1.val[0];
            g1 = pixel1.val[1];
            r1 = pixel1.val[2];
            if(abs(b0-b1) > TOLERANCE ||
               abs(g0-g1) > TOLERANCE ||
               abs(r0-r1) > TOLERANCE) count++;
        }
    }

    cout <<count<<"/"<<n0*m0<<" pixels did not match\n"
    return cout;

}

/*func: compares images and returns an image of the
 * differences
 * param: directories of the images being compared
 * output: Mat colored red where there are differences,
 * img.isempty() is true if the images werent same size
 */

Mat image_difference(string dir0, string dir1)
{
    //file extension
    string ext=".jpg";

    //open the two images
    Mat img0=imread(dir0+ext, CV_LOAD_IMAGE_GRAYSCALE);
    Mat img1=imread(dir1+ext, CV_LOAD_IMAGE_GRAYSCALE);

    //compare size
    n0 = img0.rows;
    m0 = img0.cols;
    n1 = img1.rows;
    m1 = im1.cols;
    if(n0 != n1 || m0 != m1)
    {
        cout << "image sizes did not match!\n"
        cout << "img0:("<<n0<<","<<m0<<")\n";
        cout << "img1:("<<n1<<","<<m1<<")\n";
        Mat img(0, 0, CV_8UC3, Scalar(0,0, 0));
        return img; //empty image returned
    }

    Mat img(n0, m0, CV_8UC3, Scalar(0,0, 0));
    int count = 0;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            Vec3b pixel0 = img0.at<Vec3b>(i,j);
            b0 = pixel0.val[0];
            g0 = pixel0.val[1];
            r0 = pixel0.val[2];
            Vec3b pixel1 = img1.at<Vec3b>(i,j);
            b1 = pixel1.val[0];
            g1 = pixel1.val[1];
            r1 = pixel1.val[2];
            Vec3b pixel;
            //make mismatched pixel all red
            if(abs(b0-b1) > TOLERANCE ||
               abs(g0-g1) > TOLERANCE ||
               abs(r0-r1) > TOLERANCE)
            {
                count++;
                img.at<Vec3b>(i,j).val[2] = 100;
            }
        }
    }
    cout <<count<<"/"<<n0*m0<<" pixels did not match\n"
}

void main(int argc, char **argv)
{
    int opt;
    bool batch = 0;
    bool diff = 0;
    int numComp = 0;
    string batchDir, compDir, origDir;
    int curArg = 0;
    while((opt = getopt(argc, argv, ":if:lrx")) != -1)
    {
        switch(opt)
        {
        case 'h':
            cout<<"-b <dir/to/files.txt> a list of tuples of files to compare"<<endl;
            cout<<"-n number of comparisons from batch, use with -b"<<endl;
            cout<<"-c <dir/to/compressed.jpg, not needed if -b is used>"<<endl;
            cout<<"-o <dir/to/original.jpg>,  not needed if -b is used"<<endl;
            cout<<"-d generate a difference image"<<endl;
            cout<<"-h this message"<<endl;
            break;
        case 'b':
            batchDir = argv[curArg+1]
            batch = 1;
            break;
        case 'n':
            numComp = argv[curArg+1]
            break;
        case 'c':
            if(batch) continue;
            compDir = argv[curArg+1];
            break;
        case 'o':
            if(batch) continue;
            origDir = argv[curArg+1]
            break;
        case 'd':
            diff = 1;
       }
    curArg++;
    }


    if(batch)
    {
     string cDir,oDir;
     ifstream file (batchDir)
     if(file.is_open())
     {
        int compared = 0;
        while(!file.eof() || (numComp > 0 && compared >= numComp))
        {
        getline(file, cDir);
        if(file.eof())break;
        getline(file, oDir);

        if(diff)
        {
            Mat img = image_difference(cDir, oDir);
            imwrite(cDir+"_compared.jpg", img);
        }
        else compare_count(cDir, oDir);
        }
        file.close();
     }
     else cout<<"could not open "<<batchDir<<endl
    }

    else
    {
        if(diff)
        {
            Mat img = image_difference(compDir, origDir);
            imwrite(compDir+"_compared.jpg", img);
        }
        else compare_count(compDir, origDir);
    }
}
