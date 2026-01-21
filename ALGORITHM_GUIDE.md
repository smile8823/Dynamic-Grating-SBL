# Algorithm Logic

## Background

The current algorithm application scenario is: the wave peak of a traditional spectrometer is generally assumed to be Gaussian, but in reality, there may be multiple peaks, non-standard peak shapes, multi-peak overlap, etc. For multi-frame spectral signals, it can be mathematically represented as:

Frame 1: y₁ = Φx₁ + n₁
Frame 2: y₂ = Φx₂ + n₂
...
Frame K: yₖ = Φxₖ + nₖ

Where y is the actual data, x is the signal to be solved, n is the noise, and Φ is the dictionary, which may be Gaussian or other shapes.

**Stage 1 (Offline Training & Global Parameter Estimation):** Using multi-frame historical data Y and known sparsity K, not only is the signal dictionary Φ learned, but a one-time, data-driven robust estimation of the global noise level β for the entire system is also performed.

**Stage 2 (Online Dynamic Signal Tracking):** For a continuous sampling sequence y₁, y₂, ..., yₜ, ..., using Φ and global β, combined with inter-frame information, the changes in the original signal x are tracked accurately frame by frame.

## Stage 1: Offline Training (Estimating Φ and Global β)

**Goal:**

1. Solve the optimization problem min_{Φ, X} ||Y - ΦX||²_F, where each column xᵢ of X satisfies ||xᵢ||₀ ≤ K, to obtain Φ_final.
2. Based on the final Φ and X, estimate the global noise precision β_global.

**Algorithm Adopted:** K-SVD, with an additional noise estimation step.

### Detailed Steps:

#### Steps 1.1 & 1.2: K-SVD Dictionary Learning (Same as before)

Through the alternating iteration of OMP sparse coding and SVD dictionary updates, the converged dictionary Φ_final and the corresponding multi-frame sparse representation X_final are finally obtained.

#### Step 1.3: Global Noise Precision Estimation (New)

**Principle:** After we obtain the optimal Φ_final and X_final, we consider the residual Y - Φ_final X_final that the model cannot explain as the best estimate of the noise N in the system.

**Calculation:**

a. Calculate the residual matrix: N_est = Y - Φ_final X_final.
b. Calculate the noise variance σ²: The average energy (variance) of the noise can be calculated through the Frobenius norm of the matrix.
   σ² = ||N_est||²_F / (M * D_y)
   Where M is the number of columns of Y (total number of frames), and D_y is the number of rows of Y (dimension of a single frame signal). This formula calculates the average squared error for each data point.
c. Calculate noise precision β_global: Precision is the reciprocal of variance.
   β_global = 1 / σ²

**Stage Output:** After offline training, we obtain two crucial global parameters:
*   Signal Dictionary Φ_final
*   Global Noise Precision β_global

## Stage 2: Online Dynamic Signal Tracking (Estimating x Sequence)

**Goal:** Given a continuous observation sequence y₁, y₂, ..., yₜ, ... and the outputs Φ_final and β_global from Stage 1, estimate the corresponding signal sequence x₁, x₂, ..., xₜ, ... in real-time.

**Algorithm Adopted:** A Covariance-Free SBL loop with K-sparsity constraints that utilizes time continuity for prior updates.

### Detailed Flow (In a time loop t = 1, 2, ...):

#### A. Initialization (Executed only at t=1)

*   **Initial Prior α₁:** Since there is no earlier historical information, we use an uninformative prior.
    α₁ = [1, 1, ..., 1]ᵀ
*   **Initial Signal μ₀_sparse:** Define a zero vector to start the first prior update.
    μ₀_sparse = [0, 0, ..., 0]ᵀ

#### B. Dynamic Tracking Loop (Executed for each frame yₜ)

**Step 2.1: Dynamic Prior Update**

*   **Principle:** Use the signal μ_{t-1}_sparse from the previous frame to guide the prior belief αₜ for the current frame. If a position i had a strong signal in the previous frame, we have reason to believe it is likely to exist in this frame as well, so we should reduce our "skepticism" towards it (i.e., the prior variance 1/α should increase).
*   **Calculation:**
    α_{t,i} = 1 / (μ_{t-1, sparse, i}² + ε)
    Where ε is a very small positive constant (e.g., 1e-9) to prevent division by zero and provide a "base skepticism" for signals that have never appeared.

**Step 2.2: Per-Frame SBL Estimation**

*   **Input:** Current observation yₜ, global dictionary Φ_final, dynamic prior αₜ, global noise precision β_global (as the initial β for this SBL iteration).
*   **Process:** Run the Covariance-Free SBL algorithm.
    *   **E-Step:** Use the Conjugate Gradient method to solve the linear system (βₜΦᵀΦ + diag(αₜ))μₜ = βₜΦᵀyₜ to obtain the posterior mean μₜ. Simultaneously efficiently estimate the covariance diagonal diag(Σₜ).
    *   **M-Step:** Update α and β using μₜ and diag(Σₜ). Note that β here can start from β_global and allows for fine-tuning within a single frame to adapt to local noise fluctuations.
*   **Output:** The "soft sparse" signal estimate μₜ for the current frame.

**Step 2.3: K-Sparsity Constraint Enforcement**

*   **Principle:** The output μₜ of SBL is probabilistic and may contain many small non-zero values. We use the known K (e.g., 3 grating signals) as a hard constraint to "purify" the result.
*   **Calculation:**
    a. Find the K elements with the largest absolute values in μₜ.
    b. Create a new sparse vector μₜ_sparse, keeping only these K values and setting all other elements to 0.

**Step 2.4: Output and Prepare for Next Round**

*   **Current Frame Output:** μₜ_sparse is our final signal estimate xₜ for the current frame yₜ.
*   **Preparation:** μₜ_sparse will be used in Step 2.1 of the next iteration (t+1) to generate the new dynamic prior α_{t+1}.

This loop continues continuously, achieving smooth and robust online tracking of the signal.

## Stage 1: Offline Training and Global Parameter Estimation (Detailed)

This algorithm uses multi-frame signals to find Φ and estimate the precision of noise n.
The goal of this stage is to learn a high-quality signal dictionary Φ and a global noise precision β from the multi-frame spectral data you provide.

**Input:**

*   **Y:** An M x N matrix representing multi-frame spectral data.
*   **M:** Dimension of each spectral data (e.g., 2048 wavelength/frequency points).
*   **N:** Total number of frames of training data (e.g., 1000 spectra).
*   **D:** Expected dictionary size (number of atoms). D is usually larger than M to achieve overcomplete representation (e.g., D=4096).
*   **K:** Known sparsity (e.g., the spectrum contains 3 definite signals).

**Output:**

*   **Φ_final:** An M x D dictionary matrix.
*   **β_global:** A scalar representing global noise precision.

### Detailed Implementation Steps and Pseudocode

#### Step 1.1: Initialization

**Principle:**
We need to provide an initial dictionary Φ⁽⁰⁾ for the K-SVD algorithm. A good initialization can accelerate convergence. A common practice is to randomly select D spectra from the training data Y as initial atoms and normalize them.

**Pseudocode:**

```
FUNCTION initialize_dictionary(Y, D):
    // Randomly select D indices from N columns of Y without replacement
    IF D > N:
        // If dictionary size exceeds sample size, sample with replacement or supplement with random vectors
        print("Warning: Dictionary size D > Number of samples N. Sampling with replacement.")
        selected_indices = random_sample(columns_of(Y), D, with_replacement=True)
    ELSE:
        selected_indices = random_sample(columns_of(Y), D, with_replacement=False)
    
    // Construct initial dictionary with selected columns
    Φ_initial = Y[:, selected_indices]
    
    // Normalize each column (atom) of the dictionary by L2 norm
    FOR k FROM 1 TO D:
        // φₖ is the k-th column of Φ_initial
        φₖ = Φ_initial[:, k]
        norm_φₖ = sqrt(sum(φₖ²))
        IF norm_φₖ > 0:
            Φ_initial[:, k] = φₖ / norm_φₖ
        ELSE:
            // If the selected atom is a zero vector, replace it with a random vector
            Φ_initial[:, k] = normalize(random_gaussian_vector(M))
            
    RETURN Φ_initial
```

#### Step 1.2: K-SVD Main Loop (Iterative Dictionary Learning)

**Principle:**
This is the core of Stage 1. We iteratively optimize Φ and sparse representation X by alternating between "Sparse Coding" and "Dictionary Update" steps.

**Pseudocode:**

```
FUNCTION learn_dictionary_ksvd(Y, K, D, max_iterations):
    // 1. Initialize dictionary
    Φ = initialize_dictionary(Y, D)
    X = new_matrix(D, N) // Create empty sparse representation matrix

    // 2. Iterative main loop
    FOR iter FROM 1 TO max_iterations:
        
        // ===================================
        // Step 1.2.1: Sparse Coding
        // Use OMP algorithm to find K-sparse representation xᵢ for each column yᵢ of Y
        // ===================================
        FOR i FROM 1 TO N:
            yᵢ = Y[:, i]
            xᵢ = orthogonal_matching_pursuit(yᵢ, Φ, K)
            X[:, i] = xᵢ
        
        // ===================================
        // Step 1.2.2: Dictionary Update
        // Update each column φₖ of dictionary Φ one by one
        // ===================================
        FOR k FROM 1 TO D:
            
            // a. Find signal indices where atom φₖ is used
            ωₖ = find_indices_where(X[k, :] ≠ 0)
            
            IF ωₖ is empty:
                // If the atom is never used, replace it with a new random atom
                // Or re-initialize with a data point contributing most to current error
                Φ[:, k] = ... // (Strategy for handling unused atoms)
                CONTINUE // Skip to next atom
            
            // b. Calculate error matrix Eₖ
            // Eₖ = Y_ωₖ - ∑_{j≠k} φⱼ x_ωₖ(j)ᵀ
            // Y_ωₖ is the submatrix of Y composed of columns at indices ωₖ
            // x_ωₖ(j) is the row vector of the j-th row of X at indices ωₖ
            
            // Subtract contributions of all atoms except φₖ from Y
            Y_current_residual = Y[:, ωₖ]
            FOR j FROM 1 TO D:
                IF j == k: CONTINUE
                // Broadcasting subtraction: φⱼ is a column vector, X[j, ωₖ] is a row vector
                // Outer product yields a matrix
                Y_current_residual -= outer_product(Φ[:, j], X[j, ωₖ])
            
            Eₖ = Y_current_residual
            
            // c. SVD Solution Update
            // Perform Singular Value Decomposition on error matrix Eₖ: Eₖ = U * S * Vᵀ
            U, S, V_transpose = svd(Eₖ)
            
            // d. Update dictionary atom and corresponding sparse coefficients
            // Update using left and right singular vectors corresponding to the largest singular value
            Φ[:, k] = U[:, 1]                           // First column of U
            X[k, ωₖ] = S[1, 1] * V_transpose[1, :]      // First singular value * First row of Vᵀ
            
    RETURN Φ, X // Return learned dictionary and final sparse representation
```

Pseudocode for `orthogonal_matching_pursuit` function:

```
FUNCTION orthogonal_matching_pursuit(y, Φ, K):
    // Initialization
    x = new_vector(D, zeros)        // Final sparse vector
    residual = y                    // Initial residual
    support_set = empty_set()       // Set of selected atom indices

    // Iterate K times
    FOR j FROM 1 TO K:
        // Find atom most correlated with residual
        correlations = abs(Φᵀ * residual)
        best_atom_index = argmax(correlations)
        
        // Update support set
        add best_atom_index to support_set
        
        // Solve least squares problem, update coefficients
        Φ_support = Φ[:, support_set]
        // a = (Φ_supportᵀ * Φ_support)⁻¹ * Φ_supportᵀ * y
        a = solve_least_squares(Φ_support, y) 
        
        // Update residual
        residual = y - Φ_support * a

    // Construct final sparse vector x
    x[support_set] = a
    RETURN x
```

#### Step 1.3: Global Noise Precision Estimation

**Principle:**
After K-SVD converges, we obtain the optimal Φ_final and X_final. The residual Y - ΦX that the model cannot explain is considered a realization of system noise. We calculate the average energy (variance) of this residual, and its reciprocal is the noise precision β.

```
FUNCTION estimate_global_noise_precision(Y, Φ_final, X_final):
    // 1. Calculate residual matrix
    // N_est = Y - Φ * X
    N_est = Y - dot_product(Φ_final, X_final)
    
    // 2. Calculate Sum of Squared Errors
    // SSE = ||N_est||²_F = sum(N_estᵢⱼ²)
    SSE = sum_of_squares(N_est)
    
    // 3. Get data dimensions
    M = number_of_rows(Y)
    N = number_of_columns(Y)
    
    // 4. Calculate noise variance σ²
    // σ² = SSE / (Total data points)
    sigma_squared = SSE / (M * N)
    
    // 5. Calculate noise precision β
    // Precision is the reciprocal of variance
    β_global = 1 / sigma_squared
    
    RETURN β_global
```

#### Stage 1 Complete Process Summary

```
// === Main Program Entry: Stage 1 ===
FUNCTION run_stage_one(Y, D, K, max_iterations):
    
    // Steps 1.1 & 1.2: Learn dictionary and sparse representation
    print("Starting K-SVD to learn the dictionary...")
    Φ_final, X_final = learn_dictionary_ksvd(Y, K, D, max_iterations)
    print("Dictionary learning complete.")
    
    // Step 1.3: Estimate global noise precision
    print("Estimating global noise precision...")
    β_global = estimate_global_noise_precision(Y, Φ_final, X_final)
    print(f"Global noise precision β estimated as: {β_global}")
    
    // Return final results
    RETURN Φ_final, β_global
```

At this point, the implementation idea for the first stage is very clear. You can implement it using any scientific computing library (such as Python's NumPy/SciPy/Scikit-learn) following this logical blueprint.

## Stage 2: Covariance-Free SBL Algorithm

In the second stage of signal processing, our core task is to accurately recover the hidden original signal x composed of a few grating signals from a single, potentially noisy spectral observation data y. This process is like detective work, requiring the "Modus Operandi Encyclopedia" (i.e., dictionary Φ) learned in the first stage and an assessment of the "general interference level" of the scene (i.e., global noise precision β) to solve this independent "case".

In this stage, we choose the powerful Sparse Bayesian Learning (SBL) algorithm because it perfectly balances signal sparsity and noise uncertainty. The core idea of SBL is to assign an independent "skepticism" hyperparameter αᵢ to each atom φᵢ in the dictionary. If αᵢ is small, it means we believe the corresponding signal component xᵢ is likely to exist (large variance); conversely, if αᵢ is large, it means we firmly believe xᵢ is basically zero. The algorithm iteratively updates these αᵢ, eventually automatically "pruning" most irrelevant atoms, leaving only those that truly constitute the signal, thus achieving sparse recovery.

In each iteration of SBL, calculating the posterior covariance matrix Σ is a crucial step. Although our ultimate goal is the signal mean μ, the diagonal elements Σᵢᵢ of the covariance matrix contain the uncertainty of our estimation for the current signal component μᵢ. This uncertainty is the key basis for updating "skepticism" αᵢ: the update rule αᵢ_new = 1 / (μᵢ² + Σᵢᵢ) clearly states that a new αᵢ depends not only on the magnitude of the current signal component (μᵢ²) but also on how confident we are in this estimate (Σᵢᵢ). It is this mechanism that makes SBL extremely robust in the presence of noise, avoiding the "wrong selection" problem that greedy algorithms may encounter.

To achieve efficient computation, we adopt the Covariance-Free technique. We do not directly calculate and store the huge D x D covariance matrix Σ, but instead obtain the posterior mean μ indirectly by solving a linear system of equations, and use methods like the Woodbury matrix identity to calculate only its diagonal elements. This allows the algorithm to run quickly in high-dimensional signal spaces, possessing the potential for real-time processing.

Finally, to achieve online dynamic wavelength tracking, we connect this process in series and introduce the assumption of "time continuity". Specifically, the final sparse signal x_{t-1} calculated in the previous frame (t-1) will be used to construct the initial "skepticism" prior αₜ for the SBL algorithm in the next frame (t). If a wavelength had a strong signal in the previous frame, then in the current frame, we will give it a higher initial "suspicion", i.e., a smaller α value. This mechanism is like adding "short-term memory" to the algorithm, enabling it to smoothly track the drift of signal wavelengths and effectively suppress mutations caused by single-frame noise, thereby realizing a dynamic wavelength online monitoring system that is both accurate and stable.

The characteristic of the algorithm in this stage is that while traditional SBL algorithms need to calculate the covariance matrix, the current algorithm does not need to calculate such a large covariance matrix.

**The "Computational Nightmare" of Standard SBL: Covariance Matrix Σ**

Recall our detective workflow:

E-Step Step 1: Σ = (βΦᵀΦ + A)⁻¹
E-Step Step 2: μ = βΣΦᵀy
M-Step: αᵢ_new = 1 / (μᵢ² + Σᵢᵢ)

The bottleneck of this flow lies in Step 1 and Step 3, both depending on Σ:

*   **Computational Cost O(D³):** Here D is the dimension of signal x (e.g., spectrum length 2048 or higher). Calculating the inverse of a D x D matrix has a computational complexity of about O(D³). When D is large (e.g., tens of thousands or even hundreds of thousands), this amount of calculation is catastrophic, making the algorithm too slow to use.
*   **Storage Cost O(D²):** You need to store this D x D Σ matrix completely in memory. If D=100,000, you need to store 100,000 * 100,000 = 10 billion values, which is impossible on most computers.

As a paper on this topic points out, the main obstacle faced by traditional SBL algorithms in high-dimensional settings is storing and calculating this huge covariance matrix (arxiv.org).

**The "Masterstroke" of Covariance-Free SBL: Detour**

The core idea of "Covariance-Free" methods, such as CoFEM (Covariance-Free Expectation-Maximization) proposed in a paper, is: "Do we really need to know the full face of Σ? Or do we only need some specific information from it?" (arxiv.org).

Observing the formulas carefully, we find:

1.  When calculating the posterior mean μ, what we need is not Σ itself, but the result of Σ acting on a vector (βΦᵀy).
2.  When updating hyperparameters α, what we need is not the complete Σ either, but only the elements on its diagonal Σᵢᵢ.

Covariance-Free methods utilize this key observation, completely bypassing the step of calculating the inverse of Σ.

**How is it done?**

**1. Change in solving μ: From "Inversion" to "Solving Equations"**

*   **Old Method:**
    Calculate H = βΦᵀΦ + A
    Calculate Σ = H⁻¹ (Very slow)
    Calculate μ = Σ * (βΦᵀy)
*   **New Method (Covariance-Free):**
    Observe that μ = Σ * (βΦᵀy) can be rewritten as Σ⁻¹μ = βΦᵀy.
    Substituting Σ⁻¹ = βΦᵀΦ + A, we get (βΦᵀΦ + A)μ = βΦᵀy.
    This is a standard linear system of equations of the form Hx = b. We don't need to find the inverse H⁻¹ of H at all! We can use efficient iterative algorithms (such as Conjugate Gradient) to solve for μ directly. Solving a linear system of equations is usually much faster than finding the inverse of a matrix, especially when H is sparse or well-structured.

**2. Clever Calculation of Diagonal Elements Σᵢᵢ**

This is the more magical step. These methods can estimate the diagonal elements Σᵢᵢ of Σ individually without explicitly calculating the entire Σ matrix through some clever linear algebra techniques. This usually also relies on an iterative solving process similar to the Conjugate Gradient method.

### Detailed Implementation Steps and Pseudocode

#### Step 2.1: Initialization (Initialization for the Stream)

**Principle:**
Before processing the first frame of spectral data (t=1), we need to set a starting point for the dynamic process. Since there is no historical information, we adopt an "uninformative" initial setting. All elements of the α vector are equal, representing that our initial "skepticism" towards all D atoms is the same.

**Pseudocode:**

```
FUNCTION initialize_dynamic_tracking(D):
    // Initialize alpha vector, all elements to 1, representing initial variance of 1
    // αₜ ~ 1 / (μ_{t-1}² + ε)
    // When t=1, μ₀ = 0, so α₁ ≈ 1/ε. To avoid numerical issues, can set directly to 1.
    α_current = new_vector(D, filled_with=1.0)
    
    // Initialize sparse signal estimate of previous frame as a zero vector
    μ_previous_sparse = new_vector(D, zeros)
    
    RETURN α_current, μ_previous_sparse
```

#### Step 2.2: Main Loop (Processing Each Frame yₜ)

**Principle:**
This is the core loop of online processing. For each frame yₜ in the data stream, we execute a complete "Dynamic Prior Update -> SBL Estimation -> K-Sparsity Constraint" flow.

**Pseudocode:**

```
FUNCTION track_signal_stream(y_stream, Φ_final, β_global, K, ε, ...):
    
    // Initialize dynamic process
    α_t, μ_t_minus_1_sparse = initialize_dynamic_tracking(D)
    
    // Create a list to store final sparse signal estimates
    x_estimated_stream = new_list()
    
    // Process each frame of spectral data in chronological order
    FOR y_t IN y_stream:
        
        // ===============================================
        // Step 2.2.1: Dynamic Prior Update
        // Use K-sparse signal μ_{t-1}_sparse from previous frame to update current belief αₜ
        // ===============================================
        FOR i FROM 1 TO D:
            α_t[i] = 1.0 / (μ_t_minus_1_sparse[i]**2 + ε)
        
        
        // ===============================================
        // Step 2.2.2: Per-Frame Covariance-Free SBL Estimation (CoFEM-SBL)
        // ===============================================
        μ_t_soft = run_sbl_for_single_frame(y_t, Φ_final, α_t, β_global, max_sbl_iterations, sbl_tolerance)
        
        
        // ===============================================
        // Step 2.2.3: K-Sparsity Hard Constraint Enforcement
        // ===============================================
        // Find indices of K elements with largest absolute values in μ_t_soft
        top_K_indices = find_indices_of_largest_abs_values(μ_t_soft, K)
        
        // Create new sparse vector, keeping only these K values
        μ_t_sparse = new_vector(D, zeros)
        μ_t_sparse[top_K_indices] = μ_t_soft[top_K_indices]
        
        
        // ===============================================
        // Step 2.2.4: Output and Prepare for Next Round
        // ===============================================
        // Store final estimate for this frame into output stream
        add μ_t_sparse to x_estimated_stream
        
        // Update "previous frame" signal for next loop
        μ_t_minus_1_sparse = μ_t_sparse
        
    RETURN x_estimated_stream
```

Detailed pseudocode for `run_sbl_for_single_frame` function:
This function is the computational core of Stage 2, implementing the EM iteration of Covariance-Free SBL.

```
FUNCTION run_sbl_for_single_frame(y, Φ, α_initial, β_initial, max_iter, tolerance):
    
    // Initialize EM algorithm
    α = α_initial
    β = β_initial
    μ = new_vector(D, zeros) // Initialize posterior mean
    N = number_of_rows(y)
    
    // EM Iteration Loop
    FOR iter FROM 1 TO max_iter:
        
        α_old = copy(α) // Save old alpha for convergence check
        
        // -------------------
        // E-Step: Calculate posterior statistics μ and diag(Σ)
        // -------------------
        
        // a. Calculate posterior mean μ
        // Goal is to solve linear system: (β*ΦᵀΦ + diag(α))μ = β*Φᵀy
        A = create_diagonal_matrix(α)
        H = β * (Φᵀ @ Φ) + A  // '@' denotes matrix multiplication
        b = β * (Φᵀ @ y)
        
        // Use Conjugate Gradient (CG) to efficiently solve Hμ = b, avoiding inversion
        μ = conjugate_gradient_solver(H, b)
        
        // b. Calculate covariance diagonal diag(Σ)
        // Σ = (β*ΦᵀΦ + diag(α))⁻¹ = H⁻¹
        // Use Woodbury identity here, assuming N < D
        A_inv = create_diagonal_matrix(1.0 / α)
        temp_inv = inverse( (1.0/β) * identity_matrix(N) + Φ @ A_inv @ Φᵀ )
        Σ_diag = diagonal_of( A_inv - A_inv @ Φᵀ @ temp_inv @ Φ @ A_inv )

        
        // -------------------
        // M-Step: Update hyperparameters α and β
        // -------------------
        
        // a. Update α
        FOR i FROM 1 TO D:
            α[i] = 1.0 / (μ[i]**2 + Σ_diag[i])
            
        // b. Update β
        // β_new = N / ( ||y - Φμ||² + ∑ᵢ(1 - α_oldᵢ * Σᵢᵢ) )
        residual_norm_sq = sum_of_squares(y - Φ @ μ)
        trace_term = sum(1.0 - α_old * Σ_diag) // Element-wise product
        β = N / (residual_norm_sq + trace_term / β) // (Note: trace_term/β == Tr(ΣΦᵀΦ))

        
        // Check convergence
        change_in_alpha = norm(α - α_old) / norm(α_old)
        IF change_in_alpha < tolerance:
            BREAK // Converged, break loop
            
    // Return soft sparse posterior mean calculated by SBL
    RETURN μ
```

#### How Stage 1 Results are Used in Stage 2 (Summary)

*   **Φ_final (Dictionary):** This is the foundation of all calculations in Stage 2. It is used in the E-Step of SBL to construct the core matrix H = β*ΦᵀΦ + A, and in the M-Step when updating β to calculate the residual y - Φμ. It defines the feature space of the signal.
*   **β_global (Global Noise Precision):** It provides a high-quality, data-driven initial guess for the SBL algorithm for each frame. In the `run_sbl_for_single_frame` function, `β_initial` is set to `β_global`. This is much more stable than initializing β randomly, can accelerate SBL convergence, and make its results more reliable. Inside SBL, β can be fine-tuned to adapt to single-frame noise characteristics, but its starting point is globally optimal.

## Simulated Data Generation

**Core Goal:**
Generate an M x N data matrix Y, where each column yₜ represents spectral data at a time point. This data stream needs to satisfy the following characteristics:

1.  **Signal Structure:** Each spectrum yₜ is formed by the superposition of K Gaussian-like signals.
2.  **Dynamic Evolution:** From yₜ to y_{t+1}, the center positions of these K signals will undergo smooth, random drift, simulating continuous changes in real physical processes.
3.  **Controllable Noise:** Add Gaussian white noise to the clean signal.

We will build each frame of simulated data yₜ in three steps:

1.  **Define Signal Shape:** We first define a standard, unit-height Gaussian function. This will serve as the "template" for all our signals.
2.  **Generate Dynamic Parameters:** For each frame t and each signal k (from 1 to K), we need to generate its dynamic parameters: center position c_{t,k}, amplitude A_{t,k}, and width σ_{t,k}. The core lies in that the center position c_{t,k} will be generated through a **Random Walk** process, ensuring smooth drift over time.
3.  **Synthesize and Add Noise:** Sum the K Gaussian signals generated according to dynamic parameters to obtain the clean total signal y_clean, and then add a noise vector sampled from a Gaussian distribution to obtain the final simulated data yₜ.

### Detailed Implementation Steps and Pseudocode

**Input (Configurable Parameters):**

*   **M:** Spectral dimension (e.g., 2048).
*   **N:** Total number of frames to generate (e.g., 1000).
*   **K:** Number of signals per frame (e.g., 3).
*   **initial_positions:** Initial center positions of K signals (e.g., [300, 800, 1500]).
*   **amplitude_range:** Range of amplitude variation (e.g., [0.8, 1.2]).
*   **width_range:** Range of width variation (e.g., [15, 25]).
*   **drift_std:** Standard deviation controlling severity of position drift (e.g., 0.5).
*   **noise_level:** Standard deviation of noise (e.g., 0.05).

**Output:**

*   **Y:** An M x N simulated data matrix.
*   **X_true (Optional):** A K x N matrix recording the true center positions of K signals for each frame, used for subsequent evaluation of algorithm tracking accuracy.

**Pseudocode Implementation:**

**1. Main Function**

```
FUNCTION generate_simulated_data_stream(M, N, K, initial_positions, ...):
    
    // Create an M x 1 wavelength/frequency axis
    wavelength_axis = create_axis(0, M-1, M)
    
    // Initialize storage matrices
    Y = new_matrix(M, N)
    X_true = new_matrix(K, N)
    
    // Initialize current positions of K signals
    current_positions = copy(initial_positions)
    
    // Generate each frame of data in chronological order
    FOR t FROM 1 TO N:
        
        // Step 2: Generate clean signal for current frame
        y_clean_t = generate_clean_signal_frame(wavelength_axis, K, current_positions, amplitude_range, width_range)
        
        // Step 3: Add Gaussian noise
        noise = random_gaussian_vector(M, mean=0, std=noise_level)
        y_t = y_clean_t + noise
        
        // Store generated data
        Y[:, t] = y_t
        X_true[:, t] = current_positions
        
        // Step 4: Update signal positions, prepare for next frame
        current_positions = update_positions_random_walk(current_positions, drift_std)
        
    RETURN Y, X_true
```

**2. Generate Single Frame Clean Signal (Helper Function)**

```
FUNCTION generate_clean_signal_frame(axis, K, positions, amp_range, width_range):
    
    // Create an empty signal vector
    y_clean = new_vector(length(axis), zeros)
    
    // Superimpose K Gaussian signals
    FOR k FROM 1 TO K:
        
        // Randomly generate amplitude and width for each signal
        amplitude = random_uniform(amp_range.min, amp_range.max)
        width = random_uniform(width_range.min, width_range.max)
        center_position = positions[k]
        
        // Generate single signal curve using Gaussian function
        single_signal = gaussian(axis, center=center_position, amplitude=amplitude, std_dev=width)
        
        // Accumulate single signal to total signal
        y_clean = y_clean + single_signal
        
    RETURN y_clean
```

**3. Gaussian Function (Utility Function)**

```
FUNCTION gaussian(x, center, amplitude, std_dev):
    // Standard Gaussian function formula
    // G(x) = A * exp( - (x - c)² / (2 * σ²) )
    exponent = - (x - center)**2 / (2 * std_dev**2)
    RETURN amplitude * exp(exponent)
```

**4. Random Walk Position Update (Key Dynamic Component)**

```
FUNCTION update_positions_random_walk(positions, drift_std):
    
    new_positions = copy(positions)
    
    // Perform independent random walk for each signal's position
    FOR k FROM 1 TO length(positions):
        
        // Sample a small drift from a Gaussian distribution with mean 0 and std drift_std
        drift = random_gaussian(mean=0, std=drift_std)
        
        // Update position
        new_positions[k] = positions[k] + drift
        
        // (Optional) Add boundary check to prevent signal drifting out of observation range
        // IF new_positions[k] < 0: new_positions[k] = 0
        // IF new_positions[k] > M: new_positions[k] = M
        
    RETURN new_positions
```

### Simulated Data Generation Process Summary

1.  **Start:** Program starts with user-defined `initial_positions`.
2.  **Loop Generation (t=1):**
    *   Around `initial_positions`, generate K Gaussian signals with random amplitudes and widths, superimposed to get `y_clean_1`.
    *   Add Gaussian noise to get final `y_1`.
    *   Record `y_1` and true `initial_positions`.
    *   Add a small random number (from N(0, drift_std²)) to each value of `initial_positions` to get new positions `positions_2`.
3.  **Loop Generation (t=2):**
    *   Around new `positions_2`, generate K Gaussian signals again, superimpose and add noise to get `y_2`.
    *   Record `y_2` and true `positions_2`.
    *   Update positions again with random walk to get `positions_3`.
4.  **...Repeat N times.**

This framework effectively simulates the dynamic data stream you need. By adjusting core parameters `amplitude_range`, `width_range`, `drift_std`, and `noise_level`, you can generate test datasets of various difficulties and characteristics to comprehensively evaluate the performance of your G-SBL tracking algorithm.

## Stage 3: Direction Prediction Guided Multi-Signal Intelligent Tracking System

### Core Design Philosophy

Stage 3 introduces revolutionary **Direction Prediction Mechanism** and **Multi-Independent Signal Tracking Architecture** based on Stage 2, specifically addressing key challenges in practical applications:

1.  **Computational Efficiency:** Traditional methods require searching the full spectral range, resulting in huge computation.
2.  **Signal Overlap:** Multiple FBG signals may overlap during drift, making separation difficult.
3.  **Multi-Signal Independence:** The change direction and magnitude of different sensors may be completely different.
4.  **Real-time Requirements:** Industrial applications require millisecond-level response.

### System Architecture Upgrade

#### Layered Tracking Architecture

```
System Layer
├── Global Timing Synchronization
├── Cross-Signal Consistency Check
├── System-Level Health Management
└── Unified Output Interface

Atom Set Layer
├── Signal 1: Atom_Set_1 → {atom_5, atom_12}
├── Signal 2: Atom_Set_2 → {atom_23, atom_31}
├── Signal 3: Atom_Set_3 → {atom_45, atom_52}
└── Each set managed and optimized independently

Signal Layer
├── FBG1: Independent Tracking Unit
├── FBG2: Independent Tracking Unit
├── FBG3: Independent Tracking Unit
└── Each signal tracked and switched independently
```

### Core Innovation Mechanisms

#### 1. Direction Prediction Engine

**Theoretical Basis:**
*   Physical Constraints: Temperature changes have thermal inertia, stress changes have mechanical inertia.
*   Mathematical Expression: dλ/dt(t) ≈ dλ/dt(t-Δt), change direction has continuity.

**Prediction Model:**

```python
class DirectionPredictionModel:
    def __init__(self, history_length=10):
        self.offset_history = []      # History of offsets
        self.velocity_history = []    # History of change velocities
        self.acceleration_history = [] # History of accelerations
        
    def predict_next_direction(self):
        # Multi-level prediction
        # 1. First-order: Based on current velocity
        # 2. Second-order: Based on acceleration trend
        # 3. Pattern prediction: Based on periodicity recognition
        return predicted_velocity
```

**Pattern Learning Capability:**
*   Periodic pattern detection (daily temperature cycle, equipment start/stop cycle)
*   Trend pattern recognition (linear heating, exponential decay)
*   Mutation pattern learning (equipment switch, external shock)

#### 2. Intelligent Search Space Pruning

**Computation Optimization Principle:**

```
Traditional Global Search: 41nm full spectral range
Prediction Guided Search: Local search based on historical direction

Search Range = Base Range (±0.5nm) + Adaptive Expansion (Predicted Offset × 2)
Computation Reduction: 80-95%
```

**Layered Search Strategy:**
*   High Confidence (>0.9): ±0.1nm range, 0.01nm step precise search
*   Medium Confidence (0.7-0.9): ±0.25nm range, 0.05nm step search
*   Low Confidence (<0.7): ±0.5nm range, 0.1nm step search + Global candidate

#### 3. Signal Overlap Recognition and Separation

**Overlap Scenario Analysis:**

```
Scenario 1: Co-directional drift caused by temperature
- All FBGs drift towards long wavelengths simultaneously
- Relative spacing remains unchanged
- Low overlap risk

Scenario 2: Hetero-directional drift caused by strain
- FBGs at different locations have different strains
- Relative spacing changes
- High overlap risk

Scenario 3: Mixed effect complex drift
- Temperature + Strain combined action
- Each FBG changes in different direction
- Highest overlap risk
```

**Direction Guided Separation Algorithm:**

```python
def direction_guided_signal_separation():
    # Step 1: Predict independent trajectory for each signal
    for signal_id in signals:
        predicted_position = current_position + direction_model.predict_next_direction()
        prediction_confidence = direction_model.get_confidence()
        
    # Step 2: Detect signal overlap
    overlapping_pairs = detect_overlapping_signals(signal_predictions)
    
    # Step 3: Separate signals based on direction difference
    for signal_i, signal_j in overlapping_pairs:
        direction_i = get_historical_direction(signal_i)
        direction_j = get_historical_direction(signal_j)
        
        if abs(direction_i - direction_j) > threshold:
            # Significant direction difference → Separate by direction weights
            separation_weights = calculate_direction_weights(direction_i, direction_j)
            apply_separation_weights(signal_i, signal_j, separation_weights)
```

### Atom Set Update and Offset Docking

#### Global Reference System Design

```
Level 0: Physical Wavelength Reference System (Absolute Coordinate System)
├── Spectrometer Hardware Coordinate System (4101 pixels)
├── Physical Wavelength Calibration (1527.0-1568.0 nm)
└── Globally fixed and invariant

Level 1: Atom Set Reference System (Local Coordinate System)
├── Atom_Set_A: Base Position [λ1_A, λ2_A, λ3_A]
├── Atom_Set_B: Base Position [λ1_B, λ2_B, λ3_B]
└── Each set has its own local coordinate system

Level 2: Time Reference System (Dynamic Coordinate System)
├── Currently active atom set
├── Offset relative to global reference system
└── Continuous offset calculation over time
```

#### Seamless Handover Mechanism

```python
def seamless_handover_algorithm():
    # Step 1: Parallel Run Period
    # Old set continues output, new set evaluated simultaneously
    
    # Step 2: Docking Reference Establishment
    # Calculate system offset between sets at signal stable moment
    # Δλ_set = λ_B(t0) - λ_A(t0)
    
    # Step 3: Switching Execution
    # Apply offset correction, switch to new set
    
    # Step 4: Confirmation and Lock
    # Continuous multi-frame confirmation, lock new set
```

### Multi-Signal Collaborative Management

#### Asynchronous Switching Management

```python
def asynchronous_switching_management():
    # Independent switching decision for each signal
    for signal_id in signals:
        quality = assess_signal_quality(signal_id)
        if quality < threshold:
            trigger_independent_switching(signal_id)
    
    # System-level coordination (avoid simultaneous switching)
    if multiple_signals_need_switching():
        # Priority sorting, staggered switching
        prioritized_signals = prioritize_switching()
        execute_staggered_switching(prioritized_signals)
```

#### Local Dictionary Construction

```python
def build_local_dictionaries():
    local_dictionaries = {}
    for signal_id, wavelength_range in signal_regions.items():
        # Select active atoms in that wavelength range
        relevant_atoms = select_relevant_atoms(φ_global, wavelength_range)
        φ_local_i = φ_global[:, relevant_atoms]
        local_dictionaries[signal_id] = φ_local_i
    return local_dictionaries
```

## Project Structure

```
d:\Dynamic-Grating-SBL\
├── src\
│   ├── config\          # Configuration files (JSON)
│   ├── core\            # Core algorithm implementations
│   │   ├── stage1_main.py           # Stage 1: Dictionary Learning & Global Param Estimation
│   │   ├── optimized_stage2_main.py # Stage 2: Online Tracking (SBL)
│   │   ├── ultra_fast_stage3.py     # Stage 3: High-speed Tracking
│   │   └── optimized_pytorch_sbl.py # PyTorch implementation of SBL
│   ├── modules\         # Helper modules and components
│   │   ├── data_reader.py           # Data ingestion
│   │   ├── dictionary_learning.py   # Dictionary learning logic
│   │   ├── direction_prediction.py  # Drift prediction
│   │   ├── peak_detection.py        # Peak finding
│   │   ├── signal_separation.py     # Signal separation logic
│   │   ├── signal_tracker.py        # Tracking logic
│   │   ├── waveform_reconstruction.py # Waveform reconstruction
│   │   ├── atom_set_manager.py      # Atom set management
│   │   ├── intelligent_search.py    # Search space optimization
│   │   └── memory_manager.py        # Memory management
│   ├── main.py          # Main entry point
│   ├── main_with_args.py# Entry point with command line arguments
│   └── verify_waveforms.py
├── scripts\             # Utility scripts and visualizations
│   ├── integration_test.py
│   ├── stage1_single_peak_reconstruct.py
│   ├── two_stage_visualization.py
│   └── three_stage_visualization.py
├── tests\               # Unit and integration tests
├── data\                # Input data directory
├── output\              # Output results directory
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

*   **src/core/**: Contains the main logic for each stage of the algorithm.
*   **src/modules/**: Modularized components reused across different stages.
*   **src/config/**: JSON configuration files for different run scenarios.
*   **scripts/**: Scripts for visualization and specific testing scenarios.
*   **output/**: Stores simulation results, reconstructed waveforms, and logs.

