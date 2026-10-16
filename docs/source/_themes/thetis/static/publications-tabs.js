(function () {
    function initialisePublicationTabs() {
        var tabs = document.querySelectorAll('.publication-tab');
        var sections = Array.prototype.slice.call(
            document.querySelectorAll('section, div.section')
        );

        if (!tabs.length || !sections.length) {
            return;
        }

        // Sphinx renders each year and category as a separate section.
        var categorizedSections = sections.map(function (section) {
            var heading = Array.prototype.slice.call(section.children).find(function (child) {
                return /^H[1-6]$/.test(child.tagName);
            });
            if (!heading) {
                return null;
            }
            var title = heading.textContent.trim();
            var category = /^\d{4}/.test(title) ? 'papers' :
                (/^PhD, MEng/.test(title) ? 'theses' :
                    (/^Pre-prints/.test(title) ? 'preprints' : null));
            return category ? { section: section, category: category } : null;
        }).filter(Boolean);

        function activate(tab) {
            var category = tab.getAttribute('href').substring(1);
            tabs.forEach(function (item) {
                var selected = item === tab;
                item.classList.toggle('active', selected);
                item.setAttribute('aria-selected', selected ? 'true' : 'false');
            });
            categorizedSections.forEach(function (item) {
                item.section.style.display = item.category === category ? '' : 'none';
            });
        }

        tabs.forEach(function (tab) {
            tab.addEventListener('click', function (event) {
                event.preventDefault();
                activate(tab);
            });
        });

        activate(tabs[0]);
    }

    document.addEventListener('DOMContentLoaded', initialisePublicationTabs);
}());
