data {
  int<lower=1> N;
  int<lower=1> J; // number of variables
  vector[J] yy[N];
  real t;
}

transformed data{
  int<lower=0> K = 1;
}

parameters {
  real b; 
  vector<lower=0>[J] sigma_square;
  vector[J] alpha;

}

transformed parameters{
  matrix[J,K] beta;
  cov_matrix[J] Theta;
  cov_matrix[J] Marg_cov;

  beta[1, 1] = t; 
  beta[2, 1] = b; 
  Theta = diag_matrix(sigma_square);
  
  Marg_cov = beta * beta'+ Theta;
}

model {
  b ~ normal(0, 1);
  sigma_square ~ cauchy(0,1);
  for (n in 1:N) yy ~ multi_normal(alpha, Marg_cov);
}

