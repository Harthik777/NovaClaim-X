function triggerChaosEngine() {
    console.warn("⚠️ CHAOS ENGINE INITIATED: MUTATING DOM...");
    
    // Select all actionable and structural elements
    const elements = document.querySelectorAll('button, input, form, div.container, a');
    
    elements.forEach(el => {
        // 1. Destroy standard RPA syntactic hooks
        el.id = "chaos_" + Math.random().toString(36).substring(2, 10);
        el.className = "scrambled_" + Math.random().toString(36).substring(2, 6);
        
        // 2. Randomize visual placement to break coordinate-based bots
        el.style.position = "absolute";
        el.style.left = (Math.random() * 80) + "%";
        el.style.top = (Math.random() * 80) + "%";
        el.style.transform = `rotate(${(Math.random() * 15) - 7.5}deg)`;
        el.style.zIndex = Math.floor(Math.random() * 100);
        
        // 3. The Semantic Anchor (Harthik's Lifeline for Ashish)
        const innerText = el.innerText ? el.innerText.trim().toLowerCase() : "";
        if (innerText.includes("submit") || innerText.includes("claim")) {
            el.setAttribute("data-semantic-role", "claim-submit-btn");
        }
    });

    // Toggle the hidden flag so the backend logs this as a "Mutated" run
    const mutatedFlag = document.getElementById("is_mutated_flag");
    if (mutatedFlag) {
        mutatedFlag.value = "true";
    }
    
    console.log("✔️ MUTATION COMPLETE. ENVIRONMENT IS NOW ADVERSARIAL.");
}