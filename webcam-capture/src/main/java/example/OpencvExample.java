package example;

import ch.qos.logback.core.net.SyslogOutputStream;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import nu.pattern.OpenCV;
import org.apache.commons.io.FileUtils;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.core.HttpHeaders;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;

import static org.opencv.core.CvType.CV_32FC2;

public class OpencvExample {

    public static final String SAVE_PATH = "src/main/resources";
    public static final String OCR_KEY = "RGVQck1qYm1IZ29GZHJQWmpIdERNZlFqdFl6dHdrYXM=";
    public static final String REST_URI = "https://569cdd509a4c4e31bf80651af98c4b45.apigw.ntruss.com/custom/v1/2871/015a8ef25eb2f464f0bc251746946304749b818b0dda137b9041e56e4a965dbd/general";

    public static void main(String[] args) throws IOException {
        // load opencv library
        OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

//        Mat mat = Mat.eye(3, 3, CvType.CV_8UC1);
//        System.out.println("mat = " + mat.dump());

        File imageFile = new File(SAVE_PATH + "/gongcha_menu.png");
        String imagePath = imageFile.getAbsolutePath();

        // read and convert image file to base64 string
        byte[] fileContent = FileUtils.readFileToByteArray(imageFile);
        String encodedString = Base64.getEncoder().encodeToString(fileContent);
        System.out.println("Encoded String = " + encodedString);

        // prepare for Naver OCR API call
        OcrReqImage ocrReqImg = new OcrReqImage("png", "HelloWorld");
        ocrReqImg.setData(encodedString);
        ArrayList<OcrReqImage> images = new ArrayList<>();
        images.add(ocrReqImg);
        OcrRequest ocrReq = new OcrRequest("V2", "string", 0, "ko");
        ocrReq.setImages(images);
        // call Naver OCR API
        String ocrApiCallResult = callOcrApi(REST_URI, ocrReq, OCR_KEY);
        System.out.println("OCR API Call Result = " + ocrApiCallResult);
        // extract OCR text
        String ocrText = extractOcrText(ocrApiCallResult, OcrResponse.class);
        System.out.println("Extracted OCR Text = " + ocrText);
        // extract OCR fields
        ArrayList<OcrResField> ocrResFields = extractOcrFields(ocrApiCallResult, OcrResponse.class);

        // finger detection with hsv
        // convert source image to HSV
        Mat srcImage = loadImage(imagePath);
        Mat blurImage = new Mat();
        Mat hsvImage = new Mat();
        Imgproc.blur(srcImage, blurImage, new Size(7, 7));  // remove some noise
        Imgproc.cvtColor(blurImage, hsvImage, Imgproc.COLOR_BGR2HSV);
        // mask hsv image
        Scalar scalarLower = new Scalar(0, 30, 0);
        Scalar scalarUpper = new Scalar(15, 255, 255);
        Mat maskedImage = new Mat();
        Core.inRange(hsvImage, scalarLower, scalarUpper, maskedImage);
        // save masked image
        saveImage(maskedImage, SAVE_PATH + "/gongcha_menu_detected.png");

        // morphological operators
        // dilate with large element, erode with small ones
        Mat morphImage = new Mat();
        Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));
        Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));
        Imgproc.erode(maskedImage, morphImage, erodeElement);
        Imgproc.erode(maskedImage, morphImage, erodeElement);
        Imgproc.dilate(maskedImage, morphImage, dilateElement);
        Imgproc.dilate(maskedImage, morphImage, dilateElement);

        // find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(morphImage, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

        // find the finger contour
        int centerX = srcImage.width() / 2;
        int bottomY = srcImage.height();
        Point pt = new Point(centerX, bottomY - 1);
        MatOfPoint fingerCont = findFingerContour(contours, pt);

        // draw contours
        Mat contourImage = srcImage.clone();
        List<MatOfPoint> contList = new ArrayList<>();
        contList.add(fingerCont);
        Imgproc.drawContours(contourImage, contList, -1, new Scalar(0, 255, 0), 2);
        saveImage(contourImage, SAVE_PATH + "/gongcha_menu_contour.png");

//        // get convex hull
//        List<MatOfPoint> convexHull = getConvexHull(contours);
//        // draw convex hull
//        Mat convexImage = contourImage.clone();
//        Imgproc.drawContours(convexImage, convexHull, -1, new Scalar(0, 0, 255), 2);
//        saveImage(convexImage, savePath + "/gongcha_menu_convex_hull.png");

        // find a point of the fingertip
        Point fingertip = findFingertip(fingerCont);
        // find the text closest to fingertip
        double minDist = 0;
        String closestText = "";
        for (int i=0; i<ocrResFields.size(); i++) {
            OcrResField field = ocrResFields.get(i);
            // find a center of the rectangle from the ocr text
            ArrayList<Vertex> vertices = field.getBoundingPoly().getVertices();
            Point centerOfText = findCenterOfRect(vertices);

            // get distance between two points
            double distance = getDistance(fingertip, centerOfText);
            if (i == 0) {
                minDist = distance;
                closestText = field.getInferText();
            }
            if (distance < minDist) {
                minDist = distance;
                closestText = field.getInferText();
            }
        }
        System.out.println("The text the finger is pointing to is: " + closestText);
    }

    // get distance between fingertip and center point of the ocr text
    private static double getDistance(Point fingertip, Point centerOfText) {
        return Math.sqrt(Math.pow(Math.abs(fingertip.x - centerOfText.x), 2) + Math.pow(Math.abs(fingertip.y - centerOfText.y), 2));
    }

    // calculate a center point of the ocr text rectangle
    private static Point findCenterOfRect(ArrayList<Vertex> vertices) {
        Vertex ltPt = vertices.get(0);
        Vertex brPt = vertices.get(2);
        float centerX = ltPt.getX() + (brPt.getX() - ltPt.getX()) / 2;
        float centerY = ltPt.getY() + (brPt.getY() - ltPt.getY()) / 2;

        return new Point(centerX, centerY);
    }

    // find a fingertip
    // detect a point with minimum y value
    private static Point findFingertip(MatOfPoint fingerCont) {
        List<Point> fingerPoints = fingerCont.toList();
        Point minPt = new Point();
        for (int i=0; i<fingerPoints.size(); i++) {
            Point pt = fingerPoints.get(i);
            if (i == 0) minPt = pt;
            if (pt.y < minPt.y) minPt = pt;
        }
        return minPt;
    }

    // call Naver OCR API
    private static String callOcrApi(String REST_URI, OcrRequest ocrReq, String OCR_KEY) {
        Client client = ClientBuilder.newClient();
        Response res = client.target(REST_URI)
                .request(MediaType.APPLICATION_JSON)
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON)
                .header("X-OCR-SECRET", OCR_KEY)
                .post(Entity.entity(ocrReq, MediaType.APPLICATION_JSON));
        return res.readEntity(String.class);
    }

    // extract OCR fields
    private static ArrayList<OcrResField> extractOcrFields(String ocrApiCallResult, Class<OcrResponse> ocrResponseClass) throws JsonProcessingException {
        ObjectMapper mapper = new ObjectMapper();
        OcrResponse ocrRes = mapper.readValue(ocrApiCallResult, ocrResponseClass);
        ArrayList<OcrResImage> ocrResImages = ocrRes.getImages();
        ArrayList<OcrResField> ocrResFields = new ArrayList<>();
        for (OcrResImage ocrResImage : ocrResImages) {
            ocrResFields = ocrResImage.getFields();
        }
        return ocrResFields;
    }

    // extract OCR text
    private static String extractOcrText(String ocrApiCallResult, Class<OcrResponse> ocrResponseClass) throws JsonProcessingException {
        ObjectMapper mapper = new ObjectMapper();
        OcrResponse ocrRes = mapper.readValue(ocrApiCallResult, ocrResponseClass);
        ArrayList<OcrResImage> ocrResImages = ocrRes.getImages();
        ArrayList<String> ocrTextList = new ArrayList<>();
        for (OcrResImage ocrResImage : ocrResImages) {
            ArrayList<OcrResField> ocrResFields = ocrResImage.getFields();
            for (OcrResField ocrResField : ocrResFields) {
                ocrTextList.add(ocrResField.getInferText());
            }
        }
        return String.join(" ", ocrTextList);
    }

    // load image
    private static Mat loadImage(String imagePath) {
        Imgcodecs imageCodecs = new Imgcodecs();
        return imageCodecs.imread(imagePath);
    }
    // save image
    private static void saveImage(Mat imageMatrix, String targetPath) {
        Imgcodecs imgcodecs = new Imgcodecs();
        imgcodecs.imwrite(targetPath, imageMatrix);
    }

    /**
     * find a finger contour from a lot of contours <br>
     *  - determines whether the point is inside a contour
     * @param contours
     * @return
     */
    private static MatOfPoint findFingerContour(List<MatOfPoint> contours, Point pt) {
        MatOfPoint fingerCont = new MatOfPoint();
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint cont = contours.get(i);
            MatOfPoint2f con2 = new MatOfPoint2f();
            cont.convertTo(con2, CV_32FC2);
            double pointInContour = Imgproc.pointPolygonTest(con2, pt, false);
            System.out.println("src image point(center,bottom-1): (" + pt.x + "," + pt.y + ")");
            System.out.println("point is inside or on edge?: " + (pointInContour > -1.0 ? "Yes" : "No"));
            if (pointInContour > -1) {
                System.out.println("pointInCountour index: " + i);
                fingerCont = cont;
                break;
            }
        }
        return fingerCont;
    }

    // find max contour
    private static MatOfPoint findMaxContour(List<MatOfPoint> contours) {
        MatOfPoint maxContour = new MatOfPoint();
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);

            Rect rect = Imgproc.boundingRect(contour);
            if(rect.width > rect.height) continue;

            if(i == 0) maxContour = contour;
            double contourArea = Imgproc.contourArea(contour);
            double maxContourArea = Imgproc.contourArea(maxContour);
            if(contourArea > maxContourArea) maxContour = contour;
        }
        return maxContour;
    }

    // get Convex hull
    private static List<MatOfPoint> getConvexHull(List<MatOfPoint> contours) {
        // draw convex hull
        List<MatOfPoint> hullList = new ArrayList<>();
        for (int i=0; i<contours.size(); i++) {
            MatOfPoint points = contours.get(i);
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(points, hull);
            Point[] contourArray = points.toArray();
            Point[] hullPoints = new Point[hull.rows()];
            List<Integer> hullContourIdxList = hull.toList();
            for (int j = 0; j < hullContourIdxList.size(); j++) {
                hullPoints[j] = contourArray[hullContourIdxList.get(j)];
            }
            hullList.add(new MatOfPoint(hullPoints));
        }
        return hullList;
    }
}
