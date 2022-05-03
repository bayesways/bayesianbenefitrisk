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

parameters {
  vector[J] alpha[number_of_groups];
  real beta1; // 2 free elements for cont effect
  vector[Kb] beta_b; // 4 free elements for binary effect
  vector[N] zz;
  vector<lower=0>[Kc] theta;
}

transformed parameters{
  matrix[N,Kb] yy;
  vector[Kc] beta_c;
  cov_matrix[Kc] Marg_cov_cont;
  vector[Kc] alpha_c[number_of_groups];
  vector[Kb] alpha_b[number_of_groups];
  for (g in 1:number_of_groups){
    alpha_c[g,] = alpha[g,:Kc];
    alpha_b[g,] = alpha[g,(Kc+1):];
  }
  beta_c[1] = 1;
  beta_c[2] = beta1;

  Marg_cov_cont = beta_c * beta_c' + diag_matrix(theta);
  for (n in 1:N){
    yy[n, :] = to_row_vector(alpha_b[grp[n],]) + zz[n] * beta_b';
  }
}

model {
  beta1 ~ normal(0, 1);
  beta_b ~ normal(0, 1);
  theta ~ cauchy(0,1);
  for (g in 1:number_of_groups)to_vector(alpha[g,]) ~ normal(0, 10);
  zz ~ normal(0,1);
  for (n in 1:N) yc[n,] ~ multi_normal(alpha_c[grp[n],], Marg_cov_cont);
  for (j in 1:Kb) yb[, j] ~ bernoulli_logit(yy[,j]);
}

generated quantities {
   vector[J] beta;
   beta = append_row(beta_c, beta_b);
}
