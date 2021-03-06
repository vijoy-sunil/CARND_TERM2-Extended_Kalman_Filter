#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
	
	is_initialized_ = false;
	previous_timestamp_ = 0;
		 
	// initializing matrices
	P_ = MatrixXd(4, 4);
    F_ = MatrixXd(4, 4);
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);
	H_laser_ = MatrixXd(2, 4);
	Hj_ = MatrixXd(3, 4);
	
    P_ << 1, 0, 0, 0,
	     0, 1, 0, 0,
         0, 0, 1000, 0,
         0, 0, 0, 1000;
		 
    F_ << 1, 0, 0, 0,
	     0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;
		 
	//measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
		        0, 0.0225;

	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
				0, 0.0009, 0,
				0, 0, 0.09;

	H_laser_ << 1, 0, 0, 0,
				0, 1, 0, 0;
			  
	//set the acceleration noise components
	noise_ax = 9;
	noise_ay = 9;	


}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

	VectorXd x(4);
	/*****************************************************************************
	*  Initialization
	****************************************************************************/
	if (!is_initialized_) {
	/**
	  * Initialize the state ekf_.x_ with the first measurement.
	  * Create the covariance matrix.
	*/

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
	  
	  //Convert radar from polar to cartesian coordinates and initialize state.
	  float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      x << rho*cos(phi), rho*sin(phi), 0, 0;
	  
	}
	else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
	  
	  //Initialize state.
	  x << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
	}
	previous_timestamp_ = measurement_pack.timestamp_;
    // Initialize ekf_ 
    MatrixXd Q(4,4);
    ekf_.Init( x, P_, F_, 
			   H_laser_, Hj_,			   
			   R_laser_, R_radar_, 			   
			   Q ); 
	is_initialized_ = true;
	return;
	}

	/*****************************************************************************
	*  Prediction
	****************************************************************************/

	/**
	 * Update the state transition matrix F according to the new elapsed time.
	 * Update the process noise covariance matrix.
	*/
	float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
	previous_timestamp_ = measurement_pack.timestamp_;

	float dt_2 = dt * dt;
	float dt_3 = dt_2 * dt;
	float dt_4 = dt_3 * dt;

	//Modify the F matrix so that the time is integrated
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;

	//set the process covariance matrix Q
	ekf_.Q_ = MatrixXd(4, 4);
	ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
			   0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
			   dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
			   0, dt_3/2*noise_ay, 0, dt_2*noise_ay;
	ekf_.Predict();

	/*****************************************************************************
	*  Update
	****************************************************************************/

	/**
	 * Use the sensor type to perform the update step.
	 * Update the state and covariance matrices.
	*/

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
		ekf_.UpdateEKF( measurement_pack.raw_measurements_ );
	
	else
		ekf_.Update( measurement_pack.raw_measurements_ );
	
	// print the output
	cout << "x_ = " << ekf_.x_ << endl;
	cout << "P_ = " << ekf_.P_ << endl;
	}
