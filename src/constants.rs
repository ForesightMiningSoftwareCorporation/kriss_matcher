// there is no information on the value of bin size H in the paper.
// I found two possible values, 5 from the original paper of FPFH
// https://web.archive.org/web/20240906202141/
// https://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf
// and 11 from PCL https://web.archive.org/web/20240429124409/
// https://pcl.readthedocs.io/projects/tutorials/en/latest/fpfh_estimation.html
pub const HISTOGRAM_NUM_BINS: usize = 11;
