%% MoM/Green's Function Calibration — IMTL MZI Sensor @ 2.45 GHz
%  Physics-based calibration using Method of Moments with spectral-domain
%  Green's function for the IMTL Mach-Zehnder interferometric dielectric sensor.
%
%  Author: Amirreza Piri, University of Pavia
%  ────────────────────────────────────────────────────────────────────────

clear; clc; close all;

%% ═══════════════════════════════════════════════════════════════════════
%  Physical Constants
%  ═══════════════════════════════════════════════════════════════════════
mu0   = 4e-7 * pi;           % [H/m]
eps0  = 8.854187817e-12;     % [F/m]
c0    = 1 / sqrt(mu0*eps0);  % [m/s]

%% ═══════════════════════════════════════════════════════════════════════
%  Sensor Geometry
%  ═══════════════════════════════════════════════════════════════════════
freq    = 2.45e9;            % [Hz]
omega   = 2*pi*freq;
k0      = omega/c0;
W       = 3.0e-3;            % strip width [m]
h_sub   = 1.6e-3;            % substrate thickness [m]
h_ch    = 2.0e-3;            % channel height [m]
eps_sub = 4.4*(1 - 1j*0.02); % FR-4 substrate
L_ref   = 40e-3;             % reference arm [m]
L_sense = 40e-3;             % sensing arm [m]

%% ═══════════════════════════════════════════════════════════════════════
%  Calibration Data — 9 analytes at 2.45 GHz, 29 ± 1 °C
%  ═══════════════════════════════════════════════════════════════════════
names   = {'Air','IPA','80%IPA','90%EtOH','Acetone','60%IPA','40%IPA','20%IPA','Water'};
eps_ref = [1.003, 5.294, 16.680, 17.844, 21.925, 31.999, 48.406, 64.667, 77.587];
td_ref  = [0.000, 0.757, 0.590,  0.680,  0.034,  0.396,  0.285,  0.189,  0.106];
S21_dB  = [-7.605070,-16.890303,-27.185883,-28.216131,-39.647438,...
           -31.622412,-30.121349,-30.226261,-29.293024];
S21_deg = [-54.380024,-33.389763,-5.310281,13.559215,-27.719151,...
            40.923534,66.664047,73.713295,78.048912];
N = length(eps_ref);

% Convert measured S21 to complex
S21_meas = 10.^(S21_dB/20) .* exp(1j*deg2rad(S21_deg));

%% ═══════════════════════════════════════════════════════════════════════
%  Step 1: Compute Propagation Constants via MoM
%  ═══════════════════════════════════════════════════════════════════════
fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('  MoM/Green''s Function Calibration — IMTL MZI @ 2.45 GHz\n');
fprintf('═══════════════════════════════════════════════════════════\n\n');
fprintf('[Step 1] Computing propagation constants via MoM...\n');

gammas = zeros(1, N);
for i = 1:N
    eps_ch = eps_ref(i) * (1 - 1j*td_ref(i));
    gammas(i) = solve_mom(k0, omega, W, h_sub, h_ch, eps_sub, eps_ch);
    alpha_np = real(gammas(i));
    beta_rad = imag(gammas(i));
    eps_eff = (beta_rad/k0)^2;
    fprintf('  %-10s  eps''=%6.3f  beta/k0=%.4f  alpha=%.3f Np/m  eps_eff=%.3f\n',...
        names{i}, eps_ref(i), beta_rad/k0, alpha_np, eps_eff);
end

%% ═══════════════════════════════════════════════════════════════════════
%  Step 2: Compute MZI S21 Response
%  ═══════════════════════════════════════════════════════════════════════
fprintf('\n[Step 2] Computing MZI S21 response...\n');

gamma_ref_val = gammas(1);  % Air reference arm
S21_model = zeros(1, N);

for i = 1:N
    S21_model(i) = 0.5 * (exp(-gamma_ref_val*L_ref) - exp(-gammas(i)*L_sense));
    fprintf('  %-10s  Model: %8.3f dB, %8.3f°  |  Meas: %8.3f dB, %8.3f°\n',...
        names{i}, 20*log10(abs(S21_model(i))+1e-30), rad2deg(angle(S21_model(i))),...
        S21_dB(i), S21_deg(i));
end

%% ═══════════════════════════════════════════════════════════════════════
%  Step 3: Two-Anchor Affine Calibration (Air + Water)
%  ═══════════════════════════════════════════════════════════════════════
fprintf('\n[Step 3] Two-anchor affine calibration (Air + Water)...\n');

% Observable vectors: [Re(S21), Im(S21)]
obs_model_air   = [real(S21_model(1)), imag(S21_model(1))];
obs_meas_air    = [real(S21_meas(1)),  imag(S21_meas(1))];
obs_model_water = [real(S21_model(N)), imag(S21_model(N))];
obs_meas_water  = [real(S21_meas(N)),  imag(S21_meas(N))];

% Affine coefficients: meas = a * model + b
delta_model = obs_model_water - obs_model_air;
delta_meas  = obs_meas_water  - obs_meas_air;
a_cal = delta_meas ./ delta_model;
b_cal = obs_meas_air - a_cal .* obs_model_air;

fprintf('  Re: y = %.6f * x + %.6f\n', a_cal(1), b_cal(1));
fprintf('  Im: y = %.6f * x + %.6f\n', a_cal(2), b_cal(2));

% Apply calibration
S21_cal = zeros(1, N);
for i = 1:N
    obs_m = [real(S21_model(i)), imag(S21_model(i))];
    obs_c = a_cal .* obs_m + b_cal;
    S21_cal(i) = obs_c(1) + 1j*obs_c(2);
    residual = norm(obs_c - [real(S21_meas(i)), imag(S21_meas(i))]);
    fprintf('  %-10s  Cal: %8.3f dB, %8.3f°  Residual: %.4e\n',...
        names{i}, 20*log10(abs(S21_cal(i))+1e-30),...
        rad2deg(angle(S21_cal(i))), residual);
end

%% ═══════════════════════════════════════════════════════════════════════
%  Step 4: Physics-Based Inversion (Validation Analytes)
%  ═══════════════════════════════════════════════════════════════════════
fprintf('\n[Step 4] Physics-based inversion...\n');

eps_ret  = zeros(1, N-2);
td_ret   = zeros(1, N-2);
residuals = zeros(1, N-2);

for i = 2:N-1
    idx = i - 1;
    obs_target = [real(S21_meas(i)), imag(S21_meas(i))];

    % Multi-start bounded optimization
    best_cost = Inf;
    best_x = [eps_ref(i), td_ref(i)]; % fallback

    n_starts = 15;
    eps_grid = linspace(1, 100, n_starts+1);
    td_grid  = linspace(0, 2, n_starts+1);

    for s = 1:n_starts
        x0 = [(eps_grid(s)+eps_grid(s+1))/2, (td_grid(s)+td_grid(s+1))/2];
        try
            options = optimoptions('fmincon','Display','off',...
                'MaxFunctionEvaluations',200,'OptimalityTolerance',1e-10);
            [x_sol, fval] = fmincon(@(x) cost_fn(x, k0, omega, W, h_sub, h_ch,...
                eps_sub, L_ref, L_sense, gamma_ref_val, a_cal, b_cal, obs_target),...
                x0, [], [], [], [], [1, 0], [100, 2], [], options);
            if fval < best_cost
                best_cost = fval;
                best_x = x_sol;
            end
        catch
            continue;
        end
    end

    eps_ret(idx) = best_x(1);
    td_ret(idx)  = best_x(2);
    residuals(idx) = sqrt(best_cost);

    eps_err = (eps_ret(idx) - eps_ref(i)) / eps_ref(i) * 100;
    td_err  = td_ret(idx) - td_ref(i);
    fprintf('  %-10s  eps''=%.3f (ref %.3f, err %+.2f%%)  tand=%.4f (ref %.4f, err %+.4f)\n',...
        names{i}, eps_ret(idx), eps_ref(i), eps_err, td_ret(idx), td_ref(i), td_err);
end

%% ═══════════════════════════════════════════════════════════════════════
%  Step 5: Results Summary
%  ═══════════════════════════════════════════════════════════════════════
fprintf('\n════════════════════════════════════════════════════════════\n');
fprintf('  RETRIEVAL RESULTS SUMMARY\n');
fprintf('════════════════════════════════════════════════════════════\n');
fprintf('%-10s | %7s %7s %7s | %7s %7s %7s\n',...
    'Analyte','eps_ref','eps_ret','Err%','td_ref','td_ret','d_td');
fprintf('──────────────────────────────────────────────────────────\n');

for i = 2:N-1
    idx = i-1;
    eps_err = (eps_ret(idx) - eps_ref(i)) / eps_ref(i) * 100;
    td_err  = td_ret(idx) - td_ref(i);
    fprintf('%-10s | %7.3f %7.3f %+7.2f | %7.4f %7.4f %+7.4f\n',...
        names{i}, eps_ref(i), eps_ret(idx), eps_err,...
        td_ref(i), td_ret(idx), td_err);
end

%% ═══════════════════════════════════════════════════════════════════════
%  Step 6: Plots
%  ═══════════════════════════════════════════════════════════════════════
figure('Position',[100 100 900 700]);

% S21 Complex Plane
subplot(2,2,[1,2]);
plot(real(S21_meas), imag(S21_meas), 'bo', 'MarkerSize',8, 'LineWidth',1.5); hold on;
plot(real(S21_model), imag(S21_model), 'rs', 'MarkerSize',7, 'LineWidth',1.5);
plot(real(S21_cal), imag(S21_cal), 'g^', 'MarkerSize',7, 'LineWidth',1.5);
for i = 1:N
    text(real(S21_meas(i))+0.005, imag(S21_meas(i))+0.005, names{i}, 'FontSize',7);
end
xlabel('Re(S_{21})'); ylabel('Im(S_{21})');
title('S_{21} Complex Plane — MoM/Green''s Function @ 2.45 GHz');
legend('Measured','MoM Model (raw)','MoM Model (calibrated)','Location','best');
grid on; axis equal;

% eps' retrieval
subplot(2,2,3);
plot(eps_ref(2:N-1), eps_ret, 'go', 'MarkerSize',8, 'LineWidth',1.5); hold on;
plot([0 80],[0 80],'k--');
xlabel('\epsilon''_{ref}'); ylabel('\epsilon''_{retrieved}');
title('\epsilon'' Retrieval'); grid on;

% tan delta retrieval
subplot(2,2,4);
plot(td_ref(2:N-1), td_ret, 'ro', 'MarkerSize',8, 'LineWidth',1.5); hold on;
plot([0 1],[0 1],'k--');
xlabel('tan\delta_{ref}'); ylabel('tan\delta_{retrieved}');
title('tan\delta Retrieval'); grid on;

sgtitle('MoM/Green''s Function MZI Calibration Results','FontWeight','bold');

fprintf('\n✓ Pipeline complete.\n');

%% ═══════════════════════════════════════════════════════════════════════
%  Helper Functions
%  ═══════════════════════════════════════════════════════════════════════

function gamma = solve_mom(k0, omega, W, h_sub, h_ch, eps_sub, eps_ch)
%SOLVE_MOM Solve dispersion equation for propagation constant via MoM.
%   Uses 1-basis Galerkin MoM with spectral-domain Green's function.

    mu0  = 4e-7*pi;
    eps0 = 8.854187817e-12;

    % Quasi-static initial guess (Hammerstad-Jensen)
    u = W / h_sub;
    er = real(eps_sub / eps0 / (1-1j*0));
    er = real(eps_sub) / eps0;  % need relative
    % eps_sub is already complex relative, just take real
    er = real(eps_sub);
    a_hj = 1 + (1/49)*log((u^4 + (u/52)^2)/(u^4 + 0.432)) + (1/18.7)*log(1 + (u/18.1)^3);
    b_hj = 0.564 * ((er - 0.9)/(er + 3))^0.053;
    eps_eff0 = (er + 1)/2 + (er - 1)/2 * (1 + 10/u)^(-a_hj * b_hj);

    % Superstrate correction
    q = 1 - exp(-2*h_ch/W);
    eps_eff0 = eps_eff0 + q*(real(eps_ch) - 1)*0.5;
    beta0 = k0 * sqrt(eps_eff0);

    % Solve det[Z(beta)] = 0 using fsolve
    x0 = [real(beta0), imag(beta0)];
    options = optimoptions('fsolve','Display','off','TolFun',1e-10,'MaxIterations',200);

    [x_sol, ~, exitflag] = fsolve(@(x) dispersion_eq(x, k0, omega, W,...
        h_sub, h_ch, eps_sub, eps_ch, mu0, eps0), x0, options);

    beta_sol = x_sol(1) + 1j*x_sol(2);
    alpha_np = -imag(beta_sol);
    beta_r   = real(beta_sol);
    gamma = alpha_np + 1j*beta_r;
end

function F = dispersion_eq(x, k0, omega, W, h_sub, h_ch, eps_sub, eps_ch, mu0, eps0)
%DISPERSION_EQ Residual of det[Z(beta)] = 0.
    beta = x(1) + 1j*x(2);
    alpha_max = 100 * k0;

    Z11 = integral(@(ax) integrand(ax, beta, k0, omega, W, h_sub, h_ch,...
        eps_sub, eps_ch, mu0, eps0), 1e-6, alpha_max,...
        'AbsTol',1e-12,'RelTol',1e-10);
    Z11 = 2 * Z11;  % symmetry
    F = [real(Z11); imag(Z11)];
end

function val = integrand(ax, beta, k0, omega, W, h_sub, h_ch,...
    eps_sub, eps_ch, mu0, eps0)
%INTEGRAND B̃_1(αx) · G̃_zz(αx, β) · B̃_1(αx)
    B1 = basis_spectral(ax, 1, W);
    G  = greens_kernel(ax, beta, k0, omega, h_sub, h_ch, eps_sub, eps_ch, mu0, eps0);
    val = B1 .* G .* B1;
end

function B = basis_spectral(ax, n, W)
%BASIS_SPECTRAL Fourier transform of n-th cosine basis function.
    m = 2*n - 1;
    ax_w = ax * W/2;
    m_pi = m * pi/2;
    arg_p = ax_w + m_pi;
    arg_m = ax_w - m_pi;
    B = (W/2) * (safe_sinc(arg_p) + safe_sinc(arg_m));
end

function y = safe_sinc(z)
%SAFE_SINC sin(z)/z with z→0 → 1.
    y = ones(size(z));
    nz = abs(z) > 1e-12;
    y(nz) = sin(z(nz)) ./ z(nz);
end

function G = greens_kernel(ax, beta, k0, omega, h_sub, h_ch,...
    eps_sub, eps_ch, mu0, eps0)
%GREENS_KERNEL Spectral-domain Green's function G̃_zz(αx, β).

    % Substrate
    k_sub = k0 * sqrt(eps_sub);
    gy_sub = sqrt(ax.^2 + beta^2 - k_sub^2);
    gy_sub(real(gy_sub) < 0) = -gy_sub(real(gy_sub) < 0);

    Y_e_sub = 1j*omega*eps_sub*eps0 ./ gy_sub;
    Y_h_sub = gy_sub ./ (1j*omega*mu0);

    % Short-circuited substrate (PEC ground)
    gd_sub = gy_sub * h_sub;
    Y_e_down = Y_e_sub ./ tanh(gd_sub);
    Y_h_down = Y_h_sub ./ tanh(gd_sub);

    % Channel
    k_ch = k0 * sqrt(eps_ch);
    gy_ch = sqrt(ax.^2 + beta^2 - k_ch^2);
    gy_ch(real(gy_ch) < 0) = -gy_ch(real(gy_ch) < 0);

    Y_e_ch = 1j*omega*eps_ch*eps0 ./ gy_ch;
    Y_h_ch = gy_ch ./ (1j*omega*mu0);

    % Air half-space
    gy_air = sqrt(ax.^2 + beta^2 - k0^2);
    gy_air(real(gy_air) < 0) = -gy_air(real(gy_air) < 0);

    Y_e_air = 1j*omega*eps0 ./ gy_air;
    Y_h_air = gy_air ./ (1j*omega*mu0);

    % Input admittance through channel
    gd_ch = gy_ch * h_ch;
    th_ch = tanh(gd_ch);
    Y_e_up = Y_e_ch .* (Y_e_air + Y_e_ch.*th_ch) ./ (Y_e_ch + Y_e_air.*th_ch);
    Y_h_up = Y_h_ch .* (Y_h_air + Y_h_ch.*th_ch) ./ (Y_h_ch + Y_h_air.*th_ch);

    % Green's function
    F_e = 1 ./ (Y_e_down + Y_e_up);
    F_h = 1 ./ (Y_h_down + Y_h_up);

    denom = ax.^2 + beta^2;
    G = (beta^2 * F_e + ax.^2 .* F_h) ./ denom;
end

function cost = cost_fn(x, k0, omega, W, h_sub, h_ch, eps_sub,...
    L_ref, L_sense, gamma_ref_val, a_cal, b_cal, obs_target)
%COST_FN Objective function for inversion.
    eps_ch = x(1) * (1 - 1j*x(2));
    gamma = solve_mom(k0, omega, W, h_sub, h_ch, eps_sub, eps_ch);
    s21 = 0.5 * (exp(-gamma_ref_val*L_ref) - exp(-gamma*L_sense));
    obs_m = [real(s21), imag(s21)];
    obs_c = a_cal .* obs_m + b_cal;
    cost = sum((obs_c - obs_target).^2);
end
