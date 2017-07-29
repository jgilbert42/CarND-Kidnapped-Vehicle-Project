/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 100;

  std::default_random_engine gen;

  std::normal_distribution<double> N_x(x, std[0]);
  std::normal_distribution<double> N_y(y, std[1]);
  std::normal_distribution<double> N_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = N_x(gen);
    p.y = N_y(gen);
    p.theta = N_theta(gen);
    p.weight = 1;

    particles.push_back(p);
    weights.push_back(p.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  for (Particle &p : particles) {
    if (fabs(yaw_rate) > 0.00001) {
      p.x = p.x + velocity/yaw_rate*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y = p.y + velocity/yaw_rate*(cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      p.theta = p.theta + yaw_rate*delta_t;
    } else {
      p.x = p.x + velocity*delta_t*cos(p.theta);
      p.y = p.y + velocity*delta_t*sin(p.theta);
      p.theta = p.theta;
    }

    normal_distribution<double> N_x(p.x, std_pos[0]);
    normal_distribution<double> N_y(p.y, std_pos[1]);
    normal_distribution<double> N_theta(p.theta, std_pos[2]);

    p.x = N_x(gen);
    p.y = N_y(gen);
    p.theta = N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed
  // measurement and assign the observed measurement to this particular
  // landmark.
  // NOTE: this method will NOT be called by the grading code. But
  // you will probably find it useful to implement this method and use it as a
  // helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read more about this distribution here:
  // https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  //
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located according to the MAP'S coordinate system. You will
  // need to transform between the two systems.  Keep in mind that this
  // transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to
  //   implement (look at equation 3.33 http://planning.cs.uiuc.edu/node99.html)

  weights.clear();

  // loop over all particles and update weights
  for (Particle &p : particles) {
    vector<LandmarkObs> map_observations;

    for (LandmarkObs &obs : observations) {
      // translate observation from vehicle to map coordinates
      LandmarkObs map_obs;
      map_obs.x = p.x + (obs.x*cos(p.theta) - obs.y*sin(p.theta));
      map_obs.y = p.y + (obs.x*sin(p.theta) + obs.y*cos(p.theta));
      map_observations.push_back(map_obs);
    }

    for (LandmarkObs &obs : map_observations) {
      // min distance between observations and map landmarks
      double min_dist = 1000;
      for (Map::single_landmark_s l : map_landmarks.landmark_list) {
        double d = dist(obs.x, obs.y, l.x_f, l.y_f);
        if (d < min_dist) {
          obs.id = l.id_i;
          min_dist = d;
        }
      }
    }

    // calculate weights
    p.weight = 1.0;

    double f = 1/(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
    double xden = 2*pow(std_landmark[0], 2);
    double yden = 2*pow(std_landmark[1], 2);
    for (LandmarkObs obs : map_observations) {
      Map::single_landmark_s &lm = map_landmarks.landmark_list[obs.id-1];
      double xt = pow(obs.x - lm.x_f, 2)/xden;
      double yt = pow(obs.y - lm.y_f, 2)/yden;
      double w = f * exp(-(xt + yt));
      p.weight *= w;
    }

    weights.push_back(p.weight);
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine gen;
  discrete_distribution<int> distribution(weights.begin(), weights.end());

  vector<Particle> resample_particles;

  for (int i = 0; i < particles.size(); i++) {
    resample_particles.push_back(particles[distribution(gen)]);
  }

  particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
