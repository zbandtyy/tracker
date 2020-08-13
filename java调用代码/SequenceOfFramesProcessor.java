package detectmotion;


import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import spark.config.SpeedState;
import org.apache.log4j.Logger;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import spark.config.AppConfig;
import spark.type.VideoEventData;
import java.io.IOException;
import java.io.Serializable;

import java.util.Base64;
import java.util.Date;
import java.util.List;

import static org.opencv.highgui.HighGui.imshow;
import static org.opencv.highgui.HighGui.waitKey;

/**
 * @author ：tyy
 * @date ：Created in 2020/5/31 12:57
 * @description： 对连续帧的处理进行初始化,即对视频的处理进行初始化工作，并且完成整个流程
 * @modified By：
 * @version: $
 */
@NoArgsConstructor
public class SequenceOfFramesProcessor implements Serializable {
    private static final Logger logger = Logger.getLogger(SequenceOfFramesProcessor.class);
    static {
        System.out.println("loal opencv");
        System.load(AppConfig.OPENCV_LIB_FILE);
//      System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("loal opencv success");
    }
    @Getter
    int detectedFrameGap ;//设置检测的帧数间隔进行部分的识别，即识别的帧数
    @Getter
    int frameCount = 0;//处理的是序列中的第几帧
    private  final double VIDEO_FPS = 1.0/25;
    private long batchstarttime = new Date().getTime(); //处理开始时间
    private long batchendtime = 0;
    private double FPS = 20;
    @Getter
    public   OpencvMultiTracker mtracker ;//所有的sequenceOfFrameProcessor都只拥有一个MultiTracker对象
    long firstframetime = 0;//第一帧实际时间
    long lastframetime = 0;//detectedFrameGap 的实际时间
    //一个cameraID中机器中应该只有一个这种对象
    public SequenceOfFramesProcessor(SpeedState state) {

        this.detectedFrameGap = state.getDetectedFrameGap();
        this.frameCount = state.getFrameCount();
        this.mtracker = OpencvMultiTracker.fromJson(state.getMultiTracker());
    }

    public SequenceOfFramesProcessor (Integer detectedFrameGap, String iotTransformFileName) {
        this.detectedFrameGap = detectedFrameGap;
        try {
            this.mtracker = new OpencvMultiTracker(iotTransformFileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public SequenceOfFramesProcessor (Integer detectedFrameGap, Size pic, Size real){
        this.detectedFrameGap = detectedFrameGap;
        this.mtracker = new OpencvMultiTracker(pic,real);

    }
    /***
     *
     * @param eventDatas :接受到的一组数据
     */
    public  SequenceOfFramesProcessor  processFrames(List<VideoEventData> eventDatas){
        if(mtracker == null){
            return this;
        }
        int firstEnter = 0;
        for (VideoEventData ev : eventDatas) {
            mtracker.setKey(ev.getCameraId());
            Mat frame1 = ev.getMat();
          //  Imgcodecs.imwrite("/home/user/Apache/App1/output/"+frameCount+".jpg",frame1);
            Size size = mtracker.iot.getPicSize();
            logger.info("need set size " + size);
            Mat frame = new Mat();
            if(size.height != frame1.rows() || size.width != frame1.cols()){
                Imgproc.resize(frame1,frame,size);
            }else {
                frame = frame1;
            }
          //  Imgcodecs.imwrite("/home/user/Apache/App1/output/"+frameCount+"-test.jpg",frame);
            if(frame == null){
                logger.info("frame is null,what happen?");
                continue;
            }else {
                logger.info(ev.getCameraId() + ":this is "+ frameCount +  " \n\n"  );
            }
           // mtracker.drawdetectedBoundingBox(frame);
            if (frameCount % detectedFrameGap  == 0  ) {
                if(frameCount == 0) {
                    firstframetime =  lastframetime = ev.getTime();
                } else{
                    firstframetime = lastframetime;
                    lastframetime = ev.getTime();
                }
                FPS = updateFPS();
                mtracker.detectAndCorrectObjofFrame(frame);
            } else {
                mtracker.trackObjectsofFrame(frame,firstEnter == 0);
            }
            long gaptime  = lastframetime - firstframetime;
            mtracker.drawStatistic(frame,FPS);
            mtracker.drawTrackerBox(frame,gaptime );//speed count

            // convert the "matrix of bytes" into a byte array
            Mat resultFrame = new Mat();
            Imgproc.resize(frame,resultFrame,new Size(frame1.cols(),frame1.rows()));
//            Imgcodecs.imwrite("/home/user/Apache/App1/output/"+frameCount+"-"
//                    +ev.getCameraId()+"-result.jpg",resultFrame);
            byte[] data = new byte[(int) (resultFrame.total() * resultFrame.channels())];
            resultFrame.get(0, 0, data);
            ev.setData( Base64.getEncoder().encodeToString(data));
            frameCount++;
            firstEnter++;
//            imshow("processed",frame);
//            waitKey(10);
        }
        return  this;
    }

    public void processVideo(String videoName){
        //1.打开摄像头读输入，
        VideoCapture videoCapture = new VideoCapture();
        boolean isopen = videoCapture.open(videoName);
        if(isopen == false) return;
        //2.对第一帧检测到的对象进行方框标记,并且对Tracker进行初始化
        Mat frame = new Mat();
        while (videoCapture.read(frame) ) {

            //1.实际检测的位置
            if (frameCount % detectedFrameGap  == 0) {
                FPS = updateFPS();
                mtracker.detectAndCorrectObjofFrame(frame);
 //               mtracker.drawTrackerBox(frame,10);
//                mtracker.drawBoundigBox(frame);
//
//                    imshow("pppp", frame);
//                    int key = waitKey(1000000);
//                    if (key == 16) {
//                        frameCount++;
//                        continue;
//
//                    }

            } else {
                mtracker.trackObjectsofFrame(frame,false);
            }
            mtracker.drawStatistic(frame,FPS);
            mtracker.drawTrackerBox(frame,detectedFrameGap * VIDEO_FPS);//speed count
            frameCount++;
            imshow("processed",frame);
            waitKey(10);

        }
    }

    public double updateFPS(){
        batchstarttime = batchendtime;
        batchendtime = new Date().getTime();
        double bacthFPS =  detectedFrameGap*1.0 / (batchendtime  - batchstarttime)*1000 ;
        logger.info("update FPS Success");
        return bacthFPS;
    }

    @Override
    public String toString() {
        return "SequenceOfFramesProcessor{" +
                "detectedFrameGap=" + detectedFrameGap +
                ", frameCount=" + frameCount +
                ", batchstarttime=" + batchstarttime +
                ", batchendtime=" + batchendtime +
                ", FPS=" + FPS +
                ", mtracker=" + mtracker +
                ", firstframetime=" + firstframetime +
                ", lastframetime=" + lastframetime +
                '}';
    }

    public static void main(String[] args) {
        String path = Thread.currentThread().getContextClassLoader().getResource("test1.mp4").getPath();//获取资源路径
        String xmlPath =  Thread.currentThread().getContextClassLoader().getResource("multitracker/calibrate_camera_scale.json" ).getPath();//获
        System.out.println(xmlPath);
        SequenceOfFramesProcessor opencv = new SequenceOfFramesProcessor(10,xmlPath);
        opencv.processVideo(path);
    }
}
