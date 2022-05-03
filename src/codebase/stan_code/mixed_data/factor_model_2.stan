data {
  int<lower=1> N;
  int<lower=0> Kb;
  int<lower=0> Kc;
  int<lower=0, upper=1> yb[N,Kb];
  vector[Kc] yc[N];
  int<lower=0> number_of_groups;
  int<lower=1, upper=number_of_groups> grp[N];
  int<lower=1> cov_num;
}

transformed data{
  int<lower=1> J = Kb+Kc; // number of variables
  int<lower=1> K = 2; // number of factors
  vector[K] zeros_K = rep_vector(0, K);
  cov_matrix[Kb] I_Kb = diag_matrix(rep_vector(1, Kb));
  int<lower=1> mean_num = number_of_groups;
}

parameters {
  vector[J] alpha[mean_num];
  vector[1] beta_free1_pool[cov_num]; // 2 free elements for cont effect
  vector[Kb] beta_free2_pool[cov_num]; // 4 free elements for binary effect
  matrix[N,K] zz;
  vector<lower=0>[Kc] theta_pool[cov_num];
  cholesky_factor_corr[K] L_Phi[cov_num];
}

transformed parameters{
  matrix[N,J] yy;
  matrix[J,K] beta_pool[cov_num];
  matrix[J,J] betabeta_pool[cov_num];
  cov_matrix[Kc] Marg_cov_cont_pool[cov_num];
  corr_matrix[K] Phi_cov_pool[cov_num];


  for (g in 1:cov_num){
  for(j in 1:J) {
    for (k in 1:K) beta_pool[g, j,k] = 0;
  }

  beta_pool[g, 1, 1] = 1; // fix first loading to 1
  // set the free elements
  beta_pool[g, 2, 1] = beta_free1_pool[g, 1];
  beta_pool[g, (1+Kc):J, 2] = beta_free2_pool[g,];
  // betabeta =  beta * beta'; 

  Phi_cov_pool[g, ] = multiply_lower_tri_self_transpose(L_Phi[g, ]);
  betabeta_pool[g,] =  beta_pool[g, ] * to_matrix(Phi_cov_pool[g, ]) * beta_pool[g,]';
  Marg_cov_cont_pool[g,] = betabeta_pool[g, 1:Kc,1:Kc] + diag_matrix(theta_pool[g,]);
  }
  if (cov_num==1){
    for (n in 1:N) yy[n, :] = to_row_vector(alpha[grp[n],]) + zz[n,] * beta_pool[1,]';
  }
  else if (cov_num>1){
    for (n in 1:N) yy[n, :] = to_row_vector(alpha[grp[n],]) + zz[n,] * beta_pool[grp[n],]';
  }
}

model {
  for (g in 1:cov_num){
  beta_free1_pool[g,] ~ normal(0, 1);
  beta_free2_pool[g,] ~ normal(0, 1);
  theta_pool[g,] ~ cauchy(0,1);
  L_Phi[g,] ~ lkj_corr_cholesky(2);
  }
  for (g in 1:mean_num)to_vector(alpha[g,]) ~ normal(0, 10);
  if (cov_num==1){
    for (n in 1:N) to_vector(zz[n,])  ~ multi_normal_cholesky(zeros_K, L_Phi[1,]);
    for (n in 1:N) yc[n,] ~ multi_normal(alpha[grp[n],:Kc], Marg_cov_cont_pool[1,]);
  }
  else if (cov_num>1){
    for (n in 1:N) to_vector(zz[n,])  ~ multi_normal_cholesky(zeros_K, L_Phi[grp[n]]);
    for (n in 1:N) yc[n,] ~ multi_normal(alpha[grp[n],:Kc], Marg_cov_cont_pool[grp[n],]);
  }
  for (j in 1:Kb) yb[, j] ~ bernoulli_logit(yy[,Kc+j]);

}
  

generated quantities{
  matrix[J,K] beta[number_of_groups];
  matrix[Kc,K] beta_c[number_of_groups];
  matrix[Kb,K] beta_b[number_of_groups];
  cov_matrix[Kc] Marg_cov_cont[number_of_groups];
  vector[Kc] alpha_c[number_of_groups];
  vector[Kb] alpha_b[number_of_groups];
  corr_matrix[K] Phi_cov[number_of_groups];
  for (g in 1:number_of_groups){
    if (cov_num==1){
      beta[g,] = beta_pool[1,];
      Marg_cov_cont[g,] = Marg_cov_cont_pool[1,];
      Phi_cov[g, ] = Phi_cov_pool[1,];

    }
    else if (cov_num>1){
      beta[g,] = beta_pool[g,];
      Marg_cov_cont[g,] = Marg_cov_cont_pool[g,];
      Phi_cov[g, ] = Phi_cov_pool[g,];
    }
    beta_c[g,] = beta[g,:Kc,:];
    beta_b[g,] = beta[g,(Kc+1):,:];
    alpha_c[g,] = alpha[g,:Kc];
    alpha_b[g,] = alpha[g,(Kc+1):];
    }
}
