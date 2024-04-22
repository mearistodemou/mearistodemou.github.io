data {
  int<lower = 1> N;
  int<lower = 1> N_subj;
  array[N] int<lower=1, upper=N_subj> subj;
  vector[N] Y_eeg;
  vector[N] Y_RT;
  vector[N] time;
  vector[N] Y_RT1;
}

parameters {
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  vector<lower=0>[6] tau_u;
  real alpha1;
  real alpha2;
  real beta2;
  real phi2;
  real couple3;
  matrix[6, N_subj] z_u;
  cholesky_factor_corr[6] L_u;
}

transformed parameters {
matrix[N_subj,6] u;
u = transpose(diag_pre_multiply(tau_u, L_u) * z_u);
}

model {
  // priors EEG
  alpha1 ~ normal(0, 10);
  sigma1 ~ normal(0, 50);
  // priors RT
  alpha2 ~ normal(0, 10);
  beta2 ~ normal(0,10);
  sigma2 ~ normal(0, 50);
  couple3 ~ normal(0,10);
  tau_u ~ cauchy(0,2);
  to_vector(z_u) ~ std_normal();
  L_u ~ lkj_corr_cholesky(2);
  // Specify model
  Y_eeg ~ normal(alpha1 + u[subj,1], sigma1);
  Y_RT ~ normal(alpha2 + u[subj,2] + time.*(beta2+u[subj,3]) + (Y_RT1 - (alpha2 + u[subj,2])).*(phi2+u[subj,4]) + (Y_eeg - (alpha1 + u[subj,1])).*(couple3+u[subj,5]), 
              exp(sigma2 + u[subj,6]));
}

generated quantities {
corr_matrix[6] rho_u = L_u * L_u';
vector[N] log_lik;
for (i in 1:N){
log_lik[i] = normal_lpdf(Y_eeg[i] | alpha1 + u[subj[i],1], sigma1) +
normal_lpdf(Y_RT[i] | alpha2 + u[subj[i],2] + time[i].*(beta2+u[subj[i],3]) + (Y_RT1[i] - (alpha2 + u[subj[i],2])).*(phi2+u[subj[i],4]) + (Y_eeg[i] - (alpha1 + u[subj[i],1])).*(couple3+u[subj[i],5]), 
              exp(sigma2 + u[subj[i],6]));
} 
}

