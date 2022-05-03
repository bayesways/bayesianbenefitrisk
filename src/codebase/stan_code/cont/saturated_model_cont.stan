data {
  int<lower=1> N;
  int<lower=1> J;
  vector[J] yy[N];
}

transformed data{
  cov_matrix[J] I = diag_matrix(rep_vector(1, J));

}

parameters {
  cov_matrix[J] Marg_cov;
  real alpha[1,J];
  cov_matrix[J] test[7];
}


model {
  // for (j in 1:J)
  // {
  //   for (k in 1:2) to_vector(test[j,k,]) ~ normal(0,1);
  // }
  for (j in 1:7) to_matrix(test[j,]) ~ inv_wishart(J+2, I);
  Marg_cov ~ inv_wishart(J+2, I);
  for (n in 1:N) yy ~ multi_normal(to_vector(alpha[1,]), Marg_cov);
}

generated quantities{
  
}
