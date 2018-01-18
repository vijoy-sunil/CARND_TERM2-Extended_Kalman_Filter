#include "kalman_filter.h"
#define PI 3.14159265

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &Hj_in, MatrixXd &R_in, 
						MatrixXd &R_ekf_in,  MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  
  H_ = H_in;
  Hj_ = Hj_in;
  
  R_ = R_in;
  R_ekf_ = R_ekf_in;
  
  Q_ = Q_in;
  
  I_ = Eigen::MatrixXd::Identity(4,4);
}

void KalmanFilter::Predict() {
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  
    //update the state by using Kalman Filter equations
 	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	
	MatrixXd Si = S.inverse();
	MatrixXd K = P_ * Ht * Si;

	//new estimate
	x_ = x_ + (K * y);
	P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	float px = x_[0];
	float py = x_[1];
	float vx = x_[2];
	float vy = x_[3];
  
    //update the state by using Extended Kalman Filter equations
	VectorXd h(3);
	float rho = sqrt( px*px + py*py );
	float phi = atan2(py, px);
	float rho_dot =  ( px*vx + py*vy )/rho;
	
	h << rho, phi, rho_dot;

	VectorXd y = z - h;
	while ( y[1] > 2.f*PI || y[1] < -2.f*PI ) 	//Normalizing phi
	{
		if ( y[1] > 2.f*PI ) 
			y[1] -= 2.f*PI;
		if (y[1] < -2.f*PI) 
			y[1] += 2.f*PI;
	}	
	
	Hj_ = tools.CalculateJacobian(x_);
	MatrixXd Hjt = Hj_.transpose();
	
	MatrixXd S = Hj_ * P_ * Hjt + R_ekf_;
	
	MatrixXd Si = S.inverse();
	MatrixXd K = P_ * Hjt * Si;

	//new estimate
	x_ = x_ + (K * y);
	P_ = (I_ - K * Hj_) * P_;
}
