//
// Created by gaoxiang on 19-5-4.
//

#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"

DEFINE_string(config_file, "./config/default.yaml", "config file path");

int main(int argc, char **argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    // 构造函数中只是设置了vo对象的配置文件路径
    myslam::VisualOdometry::Ptr vo(
        new myslam::VisualOdometry(FLAGS_config_file));
    // vo对象的初始化,我在想这个Init函数是不是可以直接与构造函数进行合并,反正肯定要初始化
    assert(vo->Init() == true);
    // 真正开始运行SLAM程序了,通过while(1)不断处理新的帧
    vo->Run();

    return 0;
}
