data {
  int<lower=1> N;
  int<lower=0> Kb;
  int<lower=0> Kc;
  int<lower=0, upper=1> yb[N,Kb];
  vector[Kc] yc[N];
  int<lower=0> number_of_groups;
  int<lower=1, upper=number_of_groups> grp[N];
}

transformed data{
  int<lower=1> J = Kb+Kc; // number of variables
}

generated quantities {
  vector[J] alpha[number_of_groups];
  real beta1; // 2 free elements for cont effect
  vector[Kb] beta_b; // 4 free elements for binary effect
  vector<lower=0>[Kc] theta;
  vector[Kc] beta_c;
  cov_matrix[Kc] Marg_cov_cont;
  vector[Kc] alpha_c[number_of_groups];
  vector[Kb] alpha_b[number_of_groups];
  vector[J] beta;
  for (g in 1:number_of_groups){
    for(j in 1:J) alpha[g,j] = normal_rng(0, 10);
    alpha_c[g,] = alpha[g,:Kc];
    alpha_b[g,] = alpha[g,(Kc+1):];
  }

  beta1 = normal_rng(0,1);
  beta_c[1] = 1;
  beta_c[2] = beta1;

  for (k in 1:Kb) beta_b[k] = normal_rng(0,1);
  for (k in 1:Kc) theta[k] = abs(cauchy_rng(0,1));

  Marg_cov_cont = beta_c * beta_c' + diag_matrix(theta);
  beta = append_row(beta_c, beta_b);
  
}
