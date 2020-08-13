package detectmotion;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.tracking.Tracker;

import java.io.*;


/**
 * @author ：tyy
 * @date ：Created in 2020/5/1 22:29
 * @description：用于对车子进行保存，主要用于对帧的信息进行更新
 * @modified By：
 * @version: $
 */
@NoArgsConstructor
public class CarDes implements   Serializable {
    private static final long serialVersionUID = 1L;
    @Getter  @Setter
    private Rect2d pos;//检测出来的车子的位置  当前的位置;  实时位置
    @Getter @Setter
    private Rect2d  previousPos = null;//更新一批数据的速度的第一次的位置  如果更新间隔为10帧，那么就是第一次10帧
    @Getter @Setter
    private Rect2d  netxPos = null;//更新一批数据的速度的第二次的位置   第二次10帧
    @Setter @Getter
    long count ;//对当前经过车辆的标记数
    @Getter @Setter
    double speed = 0;
    @Getter @Setter
    private Double carLength ;
    @Getter @Setter
    private transient Tracker tracker; //新数据的时候可以new处理
    @Getter
    int markedLost = 0;
    @Getter @Setter
    int phase = 0;// 0 tracker阶段, 1 检测阶段
    public CarDes(Rect2d pos, long count, Tracker t, double speed) {
        this.pos = pos;
        this.tracker = t;
        this.count = count;
    }
    public void setMarkedDelete() {
        this.markedLost ++;
    }
    @Override
    public String toString() {
        return "CarDes{" +
                "pos=" + pos +
                ", count=" + count +
                ", speed=" + speed +
                ",tracker=" + tracker +
                '}';
    }
}
