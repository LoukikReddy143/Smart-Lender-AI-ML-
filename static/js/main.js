document.addEventListener("DOMContentLoaded", function() {
    // 1. Mobile Navbar Toggle
    const navToggle = document.getElementById("navToggle");
    const navLinks = document.querySelector(".nav-links");
    
    if (navToggle && navLinks) {
        navToggle.addEventListener("click", function() {
            navLinks.classList.toggle("nav-active");
            
            // Toggle hamburger icon between bars and times
            const icon = navToggle.querySelector("i");
            if (icon.classList.contains("fa-bars")) {
                icon.classList.remove("fa-bars");
                icon.classList.add("fa-xmark");
            } else {
                icon.classList.remove("fa-xmark");
                icon.classList.add("fa-bars");
            }
        });
    }

    // 2. Real-time form validations and helper enhancements
    const loanForm = document.getElementById("loanForm");
    if (loanForm) {
        const applicantIncome = document.getElementById("ApplicantIncome");
        const coapplicantIncome = document.getElementById("CoapplicantIncome");
        const loanAmount = document.getElementById("LoanAmount");
        
        // Example: Warn user if loan amount is excessively large for their combined income
        const validateIncomeDebtRatio = () => {
            const incomeVal = parseFloat(applicantIncome.value) || 0;
            const coincomeVal = parseFloat(coapplicantIncome.value) || 0;
            const totalIncome = incomeVal + coincomeVal;
            const loanVal = parseFloat(loanAmount.value) || 0; // In thousands
            
            // If loan amount is more than 6 times their annual combined income (approx ratio calculation)
            if (totalIncome > 0 && loanVal > 0) {
                const ratio = loanVal / (totalIncome * 12 / 1000); // loan / annual income in thousands
                if (ratio > 10) {
                    loanAmount.style.borderColor = "#F59E0B"; // Amber warning
                } else {
                    loanAmount.style.borderColor = ""; // Reset
                }
            }
        };

        if (applicantIncome && coapplicantIncome && loanAmount) {
            applicantIncome.addEventListener("input", validateIncomeDebtRatio);
            coapplicantIncome.addEventListener("input", validateIncomeDebtRatio);
            loanAmount.addEventListener("input", validateIncomeDebtRatio);
        }
    }
});
